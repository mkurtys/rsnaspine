import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from spine.read import read_study

class SpineDataset(Dataset):
    def __init__(self, dicom_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.dicom_dir = dicom_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dicom_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label