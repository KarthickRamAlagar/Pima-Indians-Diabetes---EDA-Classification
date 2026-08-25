"""
preprocessing.py
IN:  raw DataFrame from data_loader.load_raw_data()
OUT: cleaned DataFrame (missing values handled) and fitted StandardScaler
     for use by both train.py and evaluate.py, so training and testing
     apply identical transformations.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def mark_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert biologically-impossible zeros to NaN so they register as missing.

    IN:  raw DataFrame (zeros in ZERO_AS_MISSING_COLS mean "not recorded")
    OUT: DataFrame with those zeros replaced by NaN
    """
    df = df.copy()
    df[ZERO_AS_MISSING_COLS] = df[ZERO_AS_MISSING_COLS].replace(0, pd.NA)
    for col in ZERO_AS_MISSING_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values with the class-conditional median.

    IN:  DataFrame with NaNs (post mark_missing), must include 'Outcome'
    OUT: DataFrame with no missing values in ZERO_AS_MISSING_COLS
    """
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df.groupby("Outcome")[col].transform(lambda x: x.fillna(x.median()))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: mark_missing -> impute_missing.

    IN:  raw DataFrame
    OUT: fully cleaned DataFrame, ready for train/test split
    """
    return impute_missing(mark_missing(df))


def fit_scaler(X_train: pd.DataFrame, save_path: str = "models/scaler.pkl") -> StandardScaler:
    """Fit a StandardScaler on the training features only, and persist it.

    IN:  X_train - training feature DataFrame (no target column)
    OUT: fitted StandardScaler, also written to save_path so evaluate.py
         can load the exact same transformation used at training time
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, save_path)
    return scaler


def load_scaler(path: str = "models/scaler.pkl") -> StandardScaler:
    """IN: path to a saved scaler. OUT: the fitted StandardScaler object."""
    return joblib.load(path)
