
import pandas as pd
from pathlib import Path
from .constants import condition_severity_map

def read_train_tables(dataset_path: Path):
    descriptions = pd.read_csv(dataset_path/"train_series_descriptions.csv")
    coordinates = pd.read_csv(dataset_path/"train_label_coordinates.csv")
    train = pd.read_csv(dataset_path/"train.csv")

    has_t2_stir = descriptions.sort_values(["study_id", "series_description"]).groupby("study_id")["series_description"].apply(lambda x: (x=="Sagittal T2/STIR").any())
    valid_studies = has_t2_stir[has_t2_stir].index
    descriptions = descriptions[descriptions["study_id"].isin(valid_studies)]
    train = train[train["study_id"].isin(valid_studies)]

    train_melt = train.melt(id_vars="study_id", var_name="condition_spec", value_name="severity").sort_values(["study_id", "condition_spec"])
    train_melt["severity_code"] = train_melt["severity"].map(condition_severity_map)
    train_melt["level"] = train_melt.apply(lambda x: "_".join(x["condition_spec"].rsplit("_", maxsplit=2)[1:]), axis=1)
    train_melt["condition"] = train_melt.apply(lambda x: x["condition_spec"].replace("left_", "").replace("right_", "").rsplit("_", maxsplit=2)[0], axis=1)

    for c in train.columns[1:]:
        train[c] = train[c].map(condition_severity_map)
    train.fillna(-1, inplace=True)
    for c in train.columns[1:]:
        train[c] = train[c].astype(int)

    coordinates["instance_number"] = coordinates["instance_number"].astype(int)
    coordinates["instance_number"] = coordinates["instance_number"]
    coordinates["level"] = coordinates["level"].str.lower().str.replace("/", "_")
    coordinates["condition_spec"] = coordinates.apply(lambda x: x["condition"].lower().replace(" ", "_") + "_" + x["level"], axis=1)
    coordinates["condition"] = coordinates.apply(lambda x: x["condition_spec"].replace("left_", "").replace("right_", "").rsplit("_", maxsplit=2)[0], axis=1)

    # coordinates = pd.merge(on=["study_id", "condition_level"], left=coordinates, right=train_melt, how="left")
    coordinates = pd.merge(on=["study_id", "series_id"], left=coordinates, right=descriptions, how="left")
    return descriptions, coordinates, train_melt, train
