import pydicom
import os
from pathlib import Path
import pandas as pd
from spine.read import read_study


def process_all(studies_path):
    files = [instance for study in studies_path.iterdir() for serie in study.iterdir() for instance in serie.iterdir()]
    results = ptqdm(read_meta, files, processes=8)
    df = pd.DataFrame(results)
    df.to_csv("meta.csv", index=False)

if __name__ == "__main__":
    studies_path = Path("/Users/mkurtys/datasets/spine/train_images")
    process_all(studies_path)

