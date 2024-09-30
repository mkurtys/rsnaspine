import os
import pandas as pd
import torch
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from spine.read import read_study
from spine.transforms import image_to_patient_coords_3d, patient_coords_to_image_2d
from spine.utils.heatmap import heatmap_3d_encoder
from spine.task.constants import condition_spec_to_idx

def custom_collate_fn(batch):
    def _concat_imgs_leave_other(x, key):
        if "image" in key or "heatmap" in key:
            return torch.tensor(np.concatenate(x))
        else:
            return torch.utils.data._utils.collate.default_collate(x)
    return {key: _concat_imgs_leave_other([d[key] for d in batch], key) for key in batch[0].keys()}


class SpineDataset(Dataset):
    def __init__(self, 
                 dicom_dir,
                 train:pd.DataFrame,
                 coordinates: pd.DataFrame,
                 descriptions: pd.DataFrame,
                 resize=(256,256),
                 transform=None
                 ):
        
        self.dicom_dir = dicom_dir
        self.train=train
        self.coordinates=coordinates
        self.descriptions=descriptions
        self.descriptions_compact = descriptions.sort_values(["study_id", "series_description"]).groupby("study_id").apply(lambda x: [(sid,d) for sid,d in zip(x["series_id"],x["series_description"]) ] )

        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.train.index)

    def __getitem__(self, idx):
        # Axial T2, Sagittal T1, Sagittal T2/STIR 
        train_row = self.train.iloc[idx]
        study_id = train_row["study_id"] # self.descriptions_compact.index[idx]
        series_desc = self.descriptions_compact.loc[study_id]
        
        study=read_study(root=self.dicom_dir, 
                         study_id=study_id,
                         series_ids=[s_id for s_id,_ in series_desc],
                         series_descriptions=[s_description for _,s_description in series_desc],
                         resize=self.resize
                        )
        
        saggital_t2_volume = study.get_saggital_t2_stir().volume
        depth, _, _ = saggital_t2_volume.shape
        if self.transform:
            saggital_t2_volume = self.transform(image=saggital_t2_volume.astype(np.float32))["image"]
        if self.train is None:
            return {"study_id": int(study_id),
                    "D": depth,
                    "image": saggital_t2_volume
                   }
        
        study_severity = train_row.values[1:-1].copy().astype(int)
        # 5 levels, 5 points, 3 grades = 75
        study_severity = study_severity

        series_scales = study.get_series_scales()
        study_coords = self.coordinates.query(f"study_id == {study_id}")
        study_coords = pd.merge(on="series_id", left=study_coords, right=series_scales, how="left")
        study_coords["x"] = study_coords["x"]*study_coords["scale"]
        study_coords["y"] = study_coords["y"]*study_coords["scale"]

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
        
        
        if len(study_coords) > 0:
            study_coords["world"] = study_coords.apply(_world_coords, axis=1)
            axial_t2_backproject = study_coords["world"].apply(lambda x: _backproject_to_image_coords(x, study.get_axial_t2()))
            study_coords["axial_t2"] = axial_t2_backproject
            study_coords["saggital_t2_stir"] = study_coords["world"].apply(lambda x: _backproject_to_image_coords(x, study.get_saggital_t2_stir()))
            study_coords["condition_spec_idx"] = study_coords["condition_spec"].map(condition_spec_to_idx)

        saggital_t2_coords_xy = np.zeros((25, 2), dtype=np.float32)
        saggital_t2_coords_z = np.zeros(25, dtype=np.float32)
        saggital_t2_coords_mask = np.zeros(25, dtype=int)
        
        world_coords = np.zeros((25, 3), dtype=np.float32)
        world_coords_mask = np.zeros(25, dtype=int)
        for i, row in study_coords.iterrows():
            saggital_t2_coords_xy[row["condition_spec_idx"]] = row["saggital_t2_stir"][0][:2]
            saggital_t2_coords_z[row["condition_spec_idx"]] = row["saggital_t2_stir"][-1]
            saggital_t2_coords_mask[row["condition_spec_idx"]] = 1
            world_coords_mask[row["condition_spec_idx"]] = 1
            world_coords[row["condition_spec_idx"]] = row["world"][::-1] #x,y,z -> z,y,x
        saggital_t2_coords_zyx = np.concatenate([saggital_t2_coords_xy, saggital_t2_coords_z.reshape(-1,1)], axis=1)[:,::-1]

        # print(world_coords)
        heatmap = heatmap_3d_encoder(study.get_saggital_t2_stir(),
                           stride=(4,4),
                           gt_coords=world_coords,
                           coords_mask=world_coords_mask,
                           gt_classes=study_severity,
                           num_classes=3,
                           sigma=2)   
        heatmap= torch.from_numpy(heatmap).float()
        heatmap = torch.permute(heatmap, (1,0,2,3))
        # print("heatmap shape", heatmap.shape)
        
        saggital_t2_slices_count =  study.get_saggital_t2_stir().volume.shape[0]
        middle_slice = saggital_t2_slices_count//2 + 1

        if self.transform:
            saggital_t2_volume = self.transform(image=saggital_t2_volume.astype(np.float32))["image"]
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
    
class SpinePredictDataset(Dataset):
    def __init__(self, 
                 dicom_dir,
                 descriptions: pd.DataFrame,
                 resize=(256,256),
                 transform=None
                 ):
        
        self.dicom_dir = dicom_dir
        self.descriptions=descriptions
        self.descriptions_compact = descriptions.sort_values(["study_id", "series_description"]).groupby("study_id").apply(lambda x: [(sid,d) for sid,d in zip(x["series_id"],x["series_description"]) ] )

        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.descriptions_compact.index)

    def __getitem__(self, idx):
        # Axial T2, Sagittal T1, Sagittal T2/STIR 
        study_id = self.descriptions_compact.index[idx]
        series_desc = self.descriptions_compact.iloc[idx]
        
        study=read_study(root=self.dicom_dir, 
                         study_id=study_id,
                         series_ids=[s_id for s_id,_ in series_desc],
                         series_descriptions=[s_description for _,s_description in series_desc],
                         resize=self.resize
                        )
        
        saggital_t2_volume = study.get_saggital_t2_stir().volume
        depth, _, _ = saggital_t2_volume.shape
        if self.transform:
            saggital_t2_volume = self.transform(image=saggital_t2_volume.astype(np.float32))["image"]
        return {"study_id": int(study_id),
                "D": depth,
                "image": saggital_t2_volume
                }