"""Download, clean and split the UCI Heart Disease (Cleveland) data."""
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def download_data(url: str = config.DATA_URL, dest=config.RAW_PATH):
    """Fetch the raw file from UCI if it is not already on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def load_raw(path=config.RAW_PATH) -> pd.DataFrame:
    """Read the raw file; '?' marks missing values in the original data."""
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values, make the target binary and fix dtypes.

    Missing values only occur in the categorical columns `ca` and `thal`,
    so they are filled with the mode.
    """
    df = df.copy()
    for col in df.columns[df.isna().any()]:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    # 0 = no disease, 1-4 = disease present -> binary label
    df[config.TARGET] = (df[config.TARGET] > 0).astype(int)
    for col in config.CATEGORICAL_FEATURES:
        df[col] = df[col].astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def build_clean_dataset(save: bool = True) -> pd.DataFrame:
    download_data()
    df = clean(load_raw())
    if save:
        config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.CLEAN_PATH, index=False)
    return df


def load_clean() -> pd.DataFrame:
    if not config.CLEAN_PATH.exists():
        return build_clean_dataset()
    return pd.read_csv(config.CLEAN_PATH)


def split(df: pd.DataFrame):
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
