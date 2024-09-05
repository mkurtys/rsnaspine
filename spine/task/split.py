import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold


def train_val_split(train_melt: pd.DataFrame):

    train_features=pd.DataFrame(
    {
        "max_severity_code": train_melt.groupby("study_id")["severity_code"].max().astype(int),
        "has_unknowns": train_melt.groupby("study_id")["severity_code"].apply(lambda x: x.isna().any()),
        "spinal_canal_stenosis": (train_melt.query("condition == 'spinal_canal_stenosis'").groupby("study_id")["severity_code"].max()>=1),
        "subarticular_stenosis": (train_melt.query("condition == 'subarticular_stenosis'").groupby("study_id")["severity_code"].max()>=1),
        "neural_foraminal_narrowing": train_melt.query("condition == 'neural_foraminal_narrowing'").groupby("study_id")["severity_code"].max()>=1
    }
    )

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2)
    train_features_col = pd.Series("", index=train_features.index)
    for col in train_features.columns:
        train_features_col += train_features[col].astype(int).astype(str)
    train_features_col
    train_features
    # train_features["stratify"] = train_features_col
    split_df = pd.DataFrame(index=train_features.index, columns={"fold": np.nan})
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features.index, train_features_col, groups=train_features.index)):
        split_df.iloc[val_idx, 0] = fold
    return split_df