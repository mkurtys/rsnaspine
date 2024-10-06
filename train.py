import torch
import torch.nn as nn
from pathlib import Path

import random
import numpy as np
import pandas as pd
import os

import pandas as pd
import torch.utils
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from spine.read import read_study
from spine.task.split import train_val_split
from spine.transforms import image_to_patient_coords_3d, patient_coords_to_image_2d
from spine.utils.heatmap import heatmap_2d_encoder, heatmap_3d_encoder
from spine.model.model import RSNASpineLightningModule
from spine.task.tables import read_train_tables
from spine.task.dataset import SpineDataset, custom_collate_fn
from spine.task.constants import condition_severity_map, condition_spec_to_idx

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
    use_wandb = True and os.environ.get("DISABLE_WANDB") is None
    root_data_dir = str(dataset_path)
    batch_size = 8
    num_workers = 12
    num_sanity_val_steps=0
    num_sanity_train_steps=0 
    # model_checkpoint_source = ModelCheckpointSource.NO_CHECKPOINT
    # model_checkpoint_uri = "checkpoints/last-v9.ckpt"
    # True, to use Config params instead of checkpoints
    # overwrite_checkpoint_hparams=True
    epochs_count = 70
    # splits_count = 1
    warmup_lr = 1e-5
    warmup_epochs = 1
    aug_prob = 0.5
    lr=5e-4
    t_max= 30
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

conditions_unsided = {
    'neural_foraminal_narrowing',
    'subarticular_stenosis'
    'spinal_canal_stenosis'
}

transforms_train = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=Config.aug_prob),
    A.GaussNoise(var_limit=(0.01, 0.05), p=Config.aug_prob),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
    ], p=Config.aug_prob/2),

    # A.OneOf([
    #    A.OpticalDistortion(distort_limit=1.0),
    #    A.GridDistortion(num_steps=5, distort_limit=1.),
    #    A.ElasticTransform(alpha=3),
    #], p=Config.aug_prob),

    # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=Config.aug_prob),
    #A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    #A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=AUG_PROB),    
    #A.Normalize(mean=0.5, std=0.5)
])

#transforms_val = A.Compose([
    #A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    #A.Normalize(mean=0.5, std=0.5)
# ])


# batchify dict with default funcitons


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

    descriptions, coordinates, train_melt, train = read_train_tables(dataset_path)

    if Config.validate:
        fold_df = train_val_split(train_melt)
        train = pd.merge(on="study_id", left=train, right=fold_df, how="left")
    else:
        train["fold"] = 1

    train_spine_dataset = SpineDataset(dicom_dir=dataset_path/"train_images/",
                                    train=train.query("fold !=  0").iloc[:, :-1],
                                    coordinates=coordinates,
                                    descriptions=descriptions,
                                    resize=(224,224),
                                    transform=transforms_train
                                    )

    val_spine_dataset = SpineDataset(dicom_dir=dataset_path/"train_images/",
                                    train=train.query("fold ==  0").iloc[:, :-1],
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
        fast_dev_run=Config.debug
    )

    # trainer.fit(model, spine_data_module.train_dataloader(), spine_data_module.val_dataloader() if Config.validate else None)

    # from spine.model.enc2d3d import Enc2d3d
    # net = Enc2d3d(pretrained=True)
    #for x in spine_data_module.train_dataloader():
    #    pass
    #     net.forward(x, output_types=("loss"))

    model = RSNASpineLightningModule(lr=Config.lr, num_training_steps=1000, num_cycles=0.5)
    trainer.fit(model, spine_data_module.train_dataloader(), spine_data_module.val_dataloader() if Config.validate else None)

        #print(x[0].shape)
        #print(type(x[0]))
