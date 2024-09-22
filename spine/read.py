import os
import pydicom
import numpy as np
import glob
import pandas as pd
from typing import List, Tuple, Union, Optional
from pathlib import Path
import cv2
from spine.spine_exam import SpineStudy, SpineSeries, InstanceMeta
from spine.transforms import normalise_to_8bit, normalise_to_01
from spine.image_utils import crop_or_pad, resize_image_with_pad


def read_instance(path):
    return pydicom.dcmread(path)


def resolve_path(root: str, study_id: str, series_id: str) -> str:
    return os.path.join(root, study_id, series_id)

# discontinous volume will be splitted to continous chunks
def read_series(
        path: os.PathLike,
        study_id: str,
        series_id: str,
        series_description: str,
        resize: Optional[Tuple[int, int]] = None
    ):
    dicom_files = glob.glob( f'{path}/*.dcm')
    if not dicom_files:
        raise ValueError(f'No dicom files found in {path}')
    dicom_files = sorted(dicom_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    instance_numbers  = [int(f.split('/')[-1].split('.')[0]) for f in dicom_files]
    dicom_instances = [ (instance_number,pydicom.dcmread(f)) for instance_number,f in zip(instance_numbers,dicom_files)]

    # todo shall we sort by projection?
    if  not all([d.pixel_array.shape==dicom_instances[0][1].pixel_array.shape for _,d in dicom_instances]):
        # compute median shape
        median_shape = np.median(np.array([d.pixel_array.shape for _,d in dicom_instances]), axis=0).astype(int)
        pixel_arrays = [crop_or_pad(d.pixel_array, median_shape)  for _,d in dicom_instances]
    else:
        pixel_arrays = [d.pixel_array for _,d in dicom_instances]


    instance_meta = []
    # instance_id_map = {}
    resized_pixel_arrays = []
    series_scale = 1.0
    for j,(i,d) in enumerate(dicom_instances):
        # instance_id_map[i] = j
        pa = pixel_arrays[j]
        position = np.array([float(v) for v in d.ImagePositionPatient])
        orientation = np.array([float(v) for v in d.ImageOrientationPatient])
        normal = np.cross(orientation[:3], orientation[3:])
        projection = np.sum(normal*position)
        pixel_spacing = np.array([float(v) for v in d.PixelSpacing])

        if resize is not None:
            resized_pa, scale = resize_image_with_pad(pa, resize)
            series_scale=scale
            resized_pixel_arrays.append(resized_pa)
            resized_spacing = pixel_spacing/scale
            rows= resized_pa.shape[0]
            cols = resized_pa.shape[1]
            original_rows = pa.shape[0]
            original_cols = pa.shape[1]
            instance_meta.append(
                InstanceMeta(
                    instance_number=i,
                    rows=rows,
                    cols=cols,
                    position=position,
                    orientation=orientation,
                    normal=normal,
                    projection=projection,
                    pixel_spacing=resized_spacing,
                    spacing_between_slices=float(d.SpacingBetweenSlices),
                    slice_thickness=float(d.SliceThickness),
                    # original_pixel_spacing=pixel_spacing,
                    # original_rows=original_rows,
                    # original_cols=original_cols,
                    scale=scale
                ))
        else:
            resized_pixel_arrays.append(pa)
            instance_meta.append(
                InstanceMeta(
                    instance_number=i,
                    rows=pa.shape[0],
                    cols=pa.shape[1],
                    position=position,
                    orientation=orientation,
                    normal=normal,
                    projection=projection,
                    pixel_spacing=resized_spacing,
                    spacing_between_slices=float(d.SpacingBetweenSlices),
                    slice_thickness=float(d.SliceThickness),
                    # original_pixel_spacing=pixel_spacing,
                    # original_rows=pa.shape[0],
                    # original_cols=pa.shape[1],
                    scale=1.0    
                ))


    volume = np.stack(resized_pixel_arrays, axis=0)
    volume = normalise_to_01(volume) #  normalise_to_8bit(volume)

    return SpineSeries(
        study_id=study_id,
        series_id=series_id,
        description=series_description,
        volume=volume,
        meta=instance_meta,
        scale=series_scale
    )

def read_study(root: str,
              study_id: str|int, series_ids: List[str|int],
              series_descriptions: List[str],
              resize: Optional[Tuple[int, int]] = None) -> SpineStudy:
    series = [read_series(resolve_path(root, str(study_id), str(sid)),
                          study_id, sid, description, resize) 
                          for sid, description in zip(series_ids, series_descriptions)]
    return SpineStudy(study_id=study_id, series=series, series_descriptions=series_descriptions)