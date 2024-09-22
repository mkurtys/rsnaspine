import pydicom
import os
from pathlib import Path
import pandas as pd
from typing import List


def read_meta(i_path):
    i = i_path[0]
    path = i_path[1]
    data = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "path": path,
        "instance_order": i,
        "PatientID": data.PatientID,
        "StudyInstanceUID": data.StudyInstanceUID,
        "SeriesInstanceUID": data.SeriesInstanceUID,
        "SOPInstanceUID": data.SOPInstanceUID,
        "PixelSpacing": data.PixelSpacing,
        "SliceThickness": data.SliceThickness,
        "SpacingBetweenSlices": data.SpacingBetweenSlices,
        "ImagePositionPatient": data.ImagePositionPatient,
        "ImageOrientationPatient": data.ImageOrientationPatient,
        "Rows": data.Rows,
        "Columns": data.Columns
    }


def ptqdm(func, iterable, processes=8, **kwargs):
    from tqdm import tqdm
    from multiprocessing import Pool
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(func, iterable, **kwargs), total=len(iterable)))
    return results


def process_all(studies_path):

    def _sort(files: List[Path]) -> List[Path]:
        return sorted(files, key=lambda x: int(x.name.split('/')[-1].split('.')[0])) 

    files = [(i,instance) for study in studies_path.iterdir() for serie in study.iterdir() for (i,instance) in enumerate(_sort(list(serie.iterdir())))]
    results = ptqdm(read_meta, files, processes=8)
    df = pd.DataFrame(results)
    df.to_csv("meta.csv", index=False)

if __name__ == "__main__":
    studies_path = Path("/Users/mkurtys/datasets/spine/train_images")
    process_all(studies_path)


