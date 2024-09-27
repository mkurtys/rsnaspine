import torch
import torch.nn as nn
from pathlib import Path

import random
import numpy as np
import pandas as pd
import os

import pandas as pd
import torch.utils
from torch.utils.data import Dataset, DataLoader
from spine.read import read_study
from spine.task.split import train_val_split
from spine.transforms import image_to_patient_coords_3d, patient_coords_to_image_2d
from spine.utils.heatmap import heatmap_2d_encoder, heatmap_3d_encoder


import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
import enum
import wandb
import os

def _to_path_or_none(path):
    return Path(path) if path is not None else None

dataset_path =  _to_path_or_none(os.environ.get("SPINE_DATASET")) or Path("/home/mkurtys/projects/datasets/spine")

class WandbMode:
    OFFLINE = "offline"
    ONLINE = "online"
    DISABLED = "disabled"

class WandbConfig:
    # change to disabled/offline if not needed
    mode =  WandbMode.ONLINE
    project = "rsna-spine"
    name = None
    save_dir = None
    log_model=False # "all"
    watch = None # all
    watch_log_freq = 100

class Config:
    use_wandb = False
    root_data_dir = str(dataset_path)
    batch_size = 6
    num_workers = 2
    num_sanity_val_steps=0
    num_sanity_train_steps=0 
    # model_checkpoint_source = ModelCheckpointSource.NO_CHECKPOINT
    # model_checkpoint_uri = "checkpoints/last-v9.ckpt"
    # True, to use Config params instead of checkpoints
    # overwrite_checkpoint_hparams=True
    epochs_count = 3
    # splits_count = 1
    warmup_lr = 1e-5
    warmup_epochs = 1
    lr=1e-3
    t_max= 3
    min_lr= 1e-4
    weight_decay=5e-3
    # mixed-precision
    precision=16
    accelerator="auto"   # auto, gpu or cpu
    # TODO - parallel computations
    num_devices=1
    validate=True
    debug=False

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# pl.seed_everything(0)


condition_severity_map = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

level_code_map = {
    "l1_l2": 0,
    "l2_l3": 1,
    "l3_l4": 2,
    "l4_l5": 3,
    "l5_s1": 4
}

conditions_spec_ordered = ['spinal_canal_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l1_l2', 'left_neural_foraminal_narrowing_l2_l3', 'left_neural_foraminal_narrowing_l3_l4', 'left_neural_foraminal_narrowing_l4_l5', 'left_neural_foraminal_narrowing_l5_s1', 'right_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3', 'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l2_l3', 'right_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l4_l5', 'right_subarticular_stenosis_l5_s1']
condition_spec_to_idx = {c: i for i, c in enumerate(conditions_spec_ordered)}
condition_spec_from_idx = {i: c for i, c in enumerate(conditions_spec_ordered)}


conditions_unsided = {
    'neural_foraminal_narrowing',
    'subarticular_stenosis'
    'spinal_canal_stenosis'
}


class SpineDataset(Dataset):
    def __init__(self, 
                 dicom_dir,
                 train:pd.DataFrame,
                 coordinates: pd.DataFrame,
                 descriptions: pd.DataFrame,
                 resize=(256,256),
                 transform=None,
                 target_transform=None,
                 prepare_heatmaps=False):
        
        self.dicom_dir = dicom_dir
        self.train=train
        self.coordinates=coordinates
        self.descriptions=descriptions
        self.descriptions_compact = descriptions.sort_values(["study_id", "series_description"]).groupby("study_id").apply(lambda x: [(sid,d) for sid,d in zip(x["series_id"],x["series_description"]) ] )

        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize
        self.prepare_heatmaps = prepare_heatmaps

    def __len__(self):
        return len(self.train.index)

    def __getitem__(self, idx):
        # Axial T2, Sagittal T1, Sagittal T2/STIR 
        train_row = self.train.iloc[idx, :]
        study_id = train_row["study_id"]
        series_desc = self.descriptions_compact.loc[study_id]
        # series_dict = {
        #     s_id:s_description for s_id,s_description in series
        # }
        
        study=read_study(root=self.dicom_dir, 
                         study_id=train_row["study_id"],
                         series_ids=[s_id for s_id,_ in series_desc],
                         series_descriptions=[s_description for _,s_description in series_desc],
                         resize=self.resize
                        )
        
        study_severity = train_row.values[1:-1].copy().astype(int)
        # 5 levels, 5 points, 3 grades = 75
        study_severity = study_severity
        # study_severity_array = np.zeros((25,3), dtype=int)
        # study_severity_mask = study_severity!=-1
        # study_severity_array[study_severity_mask, study_severity] = 1
        #study_severity_array[~study_severity_mask, :] = -1

        series_scales = study.get_series_scales()
        study_coords = self.coordinates.query(f"study_id == {study_id}")
        study_coords = pd.merge(on="series_id", left=study_coords, right=series_scales, how="left")
        study_coords["x"] = study_coords["x"]*study_coords["scale"]
        study_coords["y"] = study_coords["y"]*study_coords["scale"]

        if len(study_coords) == 0:
            pass

        def _world_coords(x):
            row_series = study.get_series(x["series_id"])
            row_instance = row_series.get_instance(x["instance_number"])
            return image_to_patient_coords_3d(x["x"], x["y"], row_instance.position, row_instance.orientation, row_instance.pixel_spacing)
        
        def _backproject_to_image_coords(world_coords, series):
            min_z = 100
            min_z_instance_number =  None
            for i, instance_meta in enumerate(series.meta):
                coords_2d, is_inside = patient_coords_to_image_2d(world_coords, instance_meta,
                               return_if_contains=True)
                if min_z > abs(coords_2d[2]):
                    min_z = abs(coords_2d[2])
                    min_z_instance_number = instance_meta.instance_number
                # if is_inside:
                #     return coords_2d, instance_meta.instance_number, min_z, i
                
            return coords_2d, min_z_instance_number, min_z, i
                
        study_coords["world"] = study_coords.apply(_world_coords, axis=1)
        axial_t2_backproject = study_coords["world"].apply(lambda x: _backproject_to_image_coords(x, study.get_axial_t2()))
        study_coords["axial_t2"] = axial_t2_backproject
        study_coords["saggital_t2_stir"] = study_coords["world"].apply(lambda x: _backproject_to_image_coords(x, study.get_saggital_t2_stir()))

        study_coords["condition_spec_idx"] = study_coords["condition_spec"].map(condition_spec_to_idx)

        saggital_t2_coords_xy = np.zeros((25, 2), dtype=np.float32)
        saggital_t2_coords_z = np.zeros(25, dtype=np.float32)
        saggital_t2_coords_mask = np.zeros(25, dtype=int)
        for i, row in study_coords.iterrows():
            saggital_t2_coords_xy[row["condition_spec_idx"]] = row["saggital_t2_stir"][0][:2]
            saggital_t2_coords_z[row["condition_spec_idx"]] = row["saggital_t2_stir"][-1]
            saggital_t2_coords_mask[row["condition_spec_idx"]] = 1
        saggital_t2_coords_zyx = np.concatenate([saggital_t2_coords_xy, saggital_t2_coords_z.reshape(-1,1)], axis=1)[:,::-1]
        heatmap = heatmap_3d_encoder(study.get_saggital_t2_stir(),
                           stride=(4,4), gt_coords=saggital_t2_coords_zyx,
                           gt_classes=study_severity,
                           num_classes=3,
                           sigma=1)
        
        heatmap= torch.from_numpy(heatmap).float()
        # heatmap.expand(25, -1, -1, -1)
        # heatmap (25, d, h, w) -> (d, 25, h, w)
        heatmap = torch.permute(heatmap, (1,0,2,3))
        # print("heatmap shape", heatmap.shape)
        

        saggital_t2_slices_count =  study.get_saggital_t2_stir().volume.shape[0]
        middle_slice = saggital_t2_slices_count//2 + 1

        saggital_t2_volume = study.get_saggital_t2_stir().volume
        depth, height, width = study.get_saggital_t2_stir().volume.shape


        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        d = {    "study_id": int(study_id),
                 "D": depth,
                 "image": saggital_t2_volume,
                  # "axial_t2": study.get_axial_t2().volume,
                 "saggital_t2_stir_coords_xy": saggital_t2_coords_xy,
                 "saggital_t2_stir_coords_z": saggital_t2_coords_z,
                 "saggital_t2_stir_coords_mask": saggital_t2_coords_mask,
                 "grade":  study_severity,
                 "heatmap": heatmap
        }
        return d
    


# batchify dict with default funcitons
def custom_collate_fn(batch):
    def _concat_imgs_leave_other(x, key):
        if "image" in key or "heatmap" in key:
            return torch.tensor(np.concatenate(x))
        else:
            return torch.utils.data._utils.collate.default_collate(x)
    return {key: _concat_imgs_leave_other([d[key] for d in batch], key) for key in batch[0].keys()}

class SpineDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, validation_dataset, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        return train_loader
        
    def val_dataloader(self):
        valid_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        return valid_loader

if __name__ == "__main__":

    descriptions = pd.read_csv(dataset_path/"train_series_descriptions.csv")
    coordinates = pd.read_csv(dataset_path/"train_label_coordinates.csv")
    submissions = pd.read_csv(dataset_path/"sample_submission.csv")
    train = pd.read_csv(dataset_path/"train.csv")

    print(train.columns.tolist())

    train_melt = train.melt(id_vars="study_id", var_name="condition_spec", value_name="severity").sort_values(["study_id", "condition_spec"])
    train_melt["severity_code"] = train_melt["severity"].map(condition_severity_map)
    train_melt["level"] = train_melt.apply(lambda x: "_".join(x["condition_spec"].rsplit("_", maxsplit=2)[1:]), axis=1)
    train_melt["condition"] = train_melt.apply(lambda x: x["condition_spec"].replace("left_", "").replace("right_", "").rsplit("_", maxsplit=2)[0], axis=1)

    for c in train.columns[1:]:
        train[c] = train[c].map(condition_severity_map)
    train.fillna(-1, inplace=True)
    for c in train.columns[1:]:
        train[c] = train[c].astype(int)
    print(train.dtypes)

    coordinates["instance_number"] = coordinates["instance_number"].astype(int)
    coordinates["instance_number"] = coordinates["instance_number"]
    coordinates["level"] = coordinates["level"].str.lower().str.replace("/", "_")
    coordinates["condition_spec"] = coordinates.apply(lambda x: x["condition"].lower().replace(" ", "_") + "_" + x["level"], axis=1)
    coordinates["condition"] = coordinates.apply(lambda x: x["condition_spec"].replace("left_", "").replace("right_", "").rsplit("_", maxsplit=2)[0], axis=1)

    # coordinates = pd.merge(on=["study_id", "condition_level"], left=coordinates, right=train_melt, how="left")
    coordinates = pd.merge(on=["study_id", "series_id"], left=coordinates, right=descriptions, how="left")

    if Config.validate:
        fold_df = train_val_split(train_melt)
        train = pd.merge(on="study_id", left=train, right=fold_df, how="left")
    else:
        train["fold"] = 1

    train_spine_dataset = SpineDataset(dicom_dir=dataset_path/"train_images/",
                                    train=train.query("fold !=  0"),
                                    coordinates=coordinates,
                                    descriptions=descriptions,
                                    resize=(224,224)
                                    )

    val_spine_dataset = SpineDataset(dicom_dir=dataset_path/"train_images/",
                                    train=train.query("fold ==  0"),
                                    coordinates=coordinates,
                                    descriptions=descriptions,
                                    resize=(224,224)
                                    )
    spine_data_module = SpineDataModule(train_dataset=train_spine_dataset, 
                                        validation_dataset=val_spine_dataset,
                                        batch_size=Config.batch_size,
                                        num_workers=Config.num_workers)
    pl_logger = None
    if Config.use_wandb:
        wandb.init(
            name=WandbConfig.name,
            project=WandbConfig.project
        )

        pl_logger = WandbLogger(
            name=WandbConfig.name,
            project=WandbConfig.project
            # save_dir=f"/kaggle/working/{cfg.project_id}/output/wandb_logs"
        )

    checkpoint_cb= ModelCheckpoint  (
        dirpath="checkpoints",
        monitor="val/loss" if Config.validate else "train/loss",
        filename="ckpt_epoch={epoch:02d}_loss={val/loss:.2f}" if Config.validate else "ckpt_epoch={epoch:02d}_loss={train/loss:.2f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        mode="min",
        save_last=True,
        every_n_epochs=5)

    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    #model_summary = pl.RichModelSummary(max_depth=2)

    trainer = pl.Trainer(
        max_epochs=Config.epochs_count,
        accelerator=Config.accelerator,
        devices=Config.num_devices,
        precision=Config.precision,
        logger = pl_logger,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar],
        profiler="simple",
        num_sanity_val_steps=Config.num_sanity_val_steps,
        fast_dev_run=Config.debug,
    )

    # trainer.fit(model, spine_data_module.train_dataloader(), spine_data_module.val_dataloader() if Config.validate else None)

    # from spine.model.enc2d3d import Enc2d3d
    # net = Enc2d3d(pretrained=True)
    # for x in spine_data_module.train_dataloader():
    #     net.forward(x, output_types=("loss"))

    from spine.model.model import RSNASpineLightningModule
    model = RSNASpineLightningModule(lr=Config.lr, max_steps=Config.t_max)
    trainer.fit(model, spine_data_module.train_dataloader(), spine_data_module.val_dataloader() if Config.validate else None)

        #print(x[0].shape)
        #print(type(x[0]))
