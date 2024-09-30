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
from spine.model.model import RSNASpineLightningModule
from spine.task.dataset import SpinePredictDataset, custom_collate_fn

import pytorch_lightning as pl
import os


def _to_path_or_none(path):
    return Path(path) if path is not None else None

dataset_path =  _to_path_or_none(os.environ.get("SPINE_DATASET")) or Path("/home/mkurtys/projects/datasets/spine")
checkpoint = _to_path_or_none(os.environ.get("CHECKPOINT")) or Path("checkpoints/last-v3.ckpt")

if __name__ == "__main__":
    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Read the tables
    descriptions = pd.read_csv(dataset_path/"test_series_descriptions.csv")
    model = RSNASpineLightningModule.load_from_checkpoint(checkpoint)
    model.eval()

    predict_dataset = SpinePredictDataset(
        dicom_dir=dataset_path/"test_images",
        descriptions=descriptions,
        resize=(256, 256)
    )

    predict_loader = DataLoader(
            predict_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=custom_collate_fn
    )
    trainer = pl.Trainer(
        accelerator="auto",
        precision=16,
    )
    predictions = trainer.predict(model, predict_loader)
    print(predictions)








