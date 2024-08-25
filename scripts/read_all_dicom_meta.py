import pydicom
import os
from pathlib import Path
import pandas as pd


def read_meta(path):
    data = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "path": path,
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
    files = [instance for study in studies_path.iterdir() for serie in study.iterdir() for instance in serie.iterdir()]
    results = ptqdm(read_meta, files, processes=8)
    df = pd.DataFrame(results)
    df.to_csv("meta.csv", index=False)

if __name__ == "__main__":
    studies_path = Path("/Users/mkurtys/datasets/spine/train_images")
    process_all(studies_path)


