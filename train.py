import torch
import torch.nn as nn
from pathlib import Path

import random
import numpy as np
import pandas as pd
import os

import os
import pandas as pd
from torch.utils.data import Dataset
from spine.read import read_study
from spine.task.split import train_val_split
from spine.transforms import image_to_patient_coords_3d



# import pytorch_lightning as pl
# from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
# from pytorch_lightning.loggers.wandb import WandbLogger
# import enum
# import wandb


dataset_path = Path("/Users/mkurtys/datasets/spine")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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

conditions_unsided = {
    'neural_foraminal_narrowing',
    'subarticular_stenosis'
    'spinal_canal_stenosis'
}


descriptions = pd.read_csv(dataset_path/"train_series_descriptions.csv")
coordinates = pd.read_csv(dataset_path/"train_label_coordinates.csv")
submissions = pd.read_csv(dataset_path/"sample_submission.csv")
train = pd.read_csv(dataset_path/"train.csv")

train_melt = train.melt(id_vars="study_id", var_name="condition_spec", value_name="severity").sort_values(["study_id", "condition_spec"])
train_melt["severity_code"] = train_melt["severity"].map(condition_severity_map)
train_melt["level"] = train_melt.apply(lambda x: "_".join(x["condition_spec"].rsplit("_", maxsplit=2)[1:]), axis=1)
train_melt["condition"] = train_melt.apply(lambda x: x["condition_spec"].replace("left_", "").replace("right_", "").rsplit("_", maxsplit=2)[0], axis=1)

print(train_melt["condition"].value_counts())

for c in train.columns[1:]:
    train[c] = train[c].map(condition_severity_map)

coordinates["instance_number"] = coordinates["instance_number"].astype(int)
coordinates["instance_number"] = coordinates["instance_number"] - 1
coordinates["level"] = coordinates["level"].str.lower().str.replace("/", "_")
coordinates["condition_spec"] = coordinates.apply(lambda x: x["condition"].lower().replace(" ", "_") + "_" + x["level"], axis=1)
coordinates["condition"] = coordinates.apply(lambda x: x["condition_spec"].replace("left_", "").replace("right_", "").rsplit("_", maxsplit=2)[0], axis=1)



print(coordinates.head())

# coordinates = pd.merge(on=["study_id", "condition_level"], left=coordinates, right=train_melt, how="left")
coordinates = pd.merge(on=["study_id", "series_id"], left=coordinates, right=descriptions, how="left")


fold_df = train_val_split(train_melt)
train = pd.merge(on="study_id", left=train, right=fold_df, how="left")


class SpineDataset(Dataset):
    def __init__(self, 
                 dicom_dir,
                 train:pd.DataFrame,
                 coordinates: pd.DataFrame,
                 descriptions: pd.DataFrame,
                 resize=(256,256),
                 transform=None,
                 target_transform=None):
        
        self.dicom_dir = dicom_dir
        self.train=train
        self.coordinates=coordinates
        self.descriptions=descriptions
        self.descriptions_compact = descriptions.sort_values(["study_id", "series_description"]).groupby("study_id").apply(lambda x: [(sid,d) for sid,d in zip(x["series_id"],x["series_description"]) ] )

        self.transform = transform
        self.target_transform = target_transform
        self.resize = resize

    def __len__(self):
        return len(self.img_labels)

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
        series_scales = study.get_series_scales()

        study_coords = self.coordinates.query(f"study_id == {study_id}")
        study_coords = pd.merge(on="series_id", left=study_coords, right=series_scales, how="left")
        study_coords["x"] = study_coords["x"]*study_coords["scale"]
        study_coords["y"] = study_coords["y"]*study_coords["scale"]

        def _world_coords(x):
            row_series = study.get_series(x["series_id"])
            row_instance = row_series.get_instance(x["instance_number"])
            return image_to_patient_coords_3d(x["x"], x["y"], row_instance.position, row_instance.orientation, row_instance.pixel_spacing)
        
        study_coords["world"] = study_coords.apply(_world_coords, axis=1)
        study_severity = train_row.values[1:]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return { "study_id": study_id,
                 "axial_t2": study.get_axial_t2().volume,
                 "saggital_t2_stir": study.get_saggital_t2_stir().volume,
                 "severity":  study_severity
              }

train_spine_dataset = SpineDataset(dicom_dir=dataset_path/"train_images/",
                                   train=train,
                                   coordinates=coordinates,
                                   descriptions=descriptions,
                                   resize=(256,256)
                                   )

print(train_spine_dataset[0])

