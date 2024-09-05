import numpy as np
import os
from pathlib import Path
import json
import pandas as pd
import functools
from spine.read import read_study

dataset_path = Path("/Users/mkurtys/datasets/spine")


def process_study(x):
    study_id, series_ids, series_descriptions = x[0],x[1],x[2]
    study = read_study(dataset_path/"train_images", study_id=study_id, series_ids=series_ids, series_descriptions=series_descriptions)
    series_stats = {}
    for series in study.series:
        vol = series.volume
        stats = {
                'mean': np.mean(vol),
                'median': np.median(vol),
                'std': np.std(vol),
        }
        series_stats[series.description] = stats
    return series_stats
        


def ptqdm(func, iterable, processes=8, **kwargs):
    from tqdm import tqdm
    from multiprocessing import Pool
    with Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(func, iterable, **kwargs), total=len(iterable)))
    return results

def process_all():
    descriptions = pd.read_csv(dataset_path/"train_series_descriptions.csv")
    study_args = list(descriptions.groupby("study_id").agg(list).itertuples(index=True, name=None)) 
    results = ptqdm(process_study, study_args, processes=8)
    return results

def aggregate(results):
    da = {}
    a = {}
    for study in results:
        for series_desc, series_stats in study.items():
            for k, v in series_stats.items():
                a.setdefault(series_desc, {}).setdefault(k, []).append(v)

    for series_desc, series_stats in a.items():
        da[series_desc] = \
            {
                'mean': float(np.mean(series_stats["mean"])),
                'median': float(np.median(series_stats["median"])),
                'std': float(np.mean(series_stats["std"])),
            }
    return da


if __name__ == "__main__":
    results = process_all()
    summary = aggregate(results)
    print(json.dumps(summary, indent=2))
    with open("intensivity_stats.json", "w") as f:
        json.dump(summary, f, indent=2)