import os
import pydicom
import numpy as np
import glob
import pandas as pd
from typing import List, Tuple, Union
from pathlib import Path
from spine.spine_exam import SpineStudy, SpineSeries, InstanceMeta
from spine.transforms import normalise_to_8bit


def read_instance(path):
    return pydicom.dcmread(path)


def resolve_path(root: str, study_id: str, series_id: str) -> str:
    return os.path.join(root, study_id, series_id)

# discontinous volume will be splitted to continous chunks
def read_series(
        path: os.PathLike,
        study_id: str,
        series_id: str,
        series_description: str
    ):
    dicom_files = glob.glob( f'{path}/*.dcm')
    dicom_files = sorted(dicom_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    instance_numbers  = [int(f.split('/')[-1].split('.')[0]) for f in dicom_files]
    dicom_instances = [ (instance_number,pydicom.dcmread(f)) for instance_number,f in zip(instance_numbers,dicom_files)]

    instance_meta = []
    for i,d in dicom_instances:
        position = np.array([float(v) for v in d.ImagePositionPatient])
        orientation = np.array([float(v) for v in d.ImageOrientationPatient])
        normal = np.cross(orientation[:3], orientation[3:])
        projection = np.sum(normal*position)
        instance_meta.append(
            InstanceMeta(
                instance_number=i,
                rows=int(d.Rows),
                cols=int(d.Columns),
                position=position,
                orientation=orientation,
                normal=normal,
                projection=projection,
                pixel_spacing=np.array([float(v) for v in d.PixelSpacing]),
                spacing_between_slices=float(d.SpacingBetweenSlices),
                slice_thickness=float(d.SliceThickness),
            )
        )

    # todo shall we sort by projection?
    volume = np.stack([d.pixel_array for _,d in dicom_instances])
    volume = normalise_to_8bit(volume)

    return SpineSeries(
        study_id=study_id,
        series_id=series_id,
        description=series_description,
        volume=volume,
        meta=instance_meta
    )

def read_study(root: str,
              study_id: str|int, series_ids: List[str|int],
              series_descriptions: List[str]) -> SpineStudy:
    series = [read_series(resolve_path(root, str(study_id), str(sid)),
                          study_id, sid, description) 
                          for sid, description in zip(series_ids, series_descriptions)]
    return SpineStudy(study_id=study_id, series=series)