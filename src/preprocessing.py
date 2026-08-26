"""
preprocessing.py
IN:  raw DataFrame from data_loader.load_raw_data()
OUT: cleaned DataFrame and a fitted StandardScaler, used identically by
     both train.py and evaluate.py so training and testing apply the
     same transformation.

Unlike the Pima dataset, the CDC Diabetes Health Indicators dataset has
ZERO missing values (confirmed via X.isnull().sum() on the raw fetch) -
it was already cleaned by the CDC/Kaggle before publication. So there is
no missing-value imputation step here: clean_data() is a pass-through,
kept only so the rest of the pipeline has one consistent entry point.
For Assignment 1 Q1 task 6 ("handle missing values"), the answer is
"not needed - the source dataset is pre-cleaned", not an imputation step.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pass-through: the CDC dataset has no missing values to handle.

    IN:  raw DataFrame from load_raw_data()
    OUT: the same DataFrame (kept as a function for pipeline consistency)
    """
    return df.copy()


def fit_scaler(X_train: pd.DataFrame, save_path: str = "models/scaler.pkl") -> StandardScaler:
    """Fit a StandardScaler on the training features only, and persist it.

    Scaling is still needed even though most features are already binary
    (0/1): KNN is distance-based, and unscaled ordinal features like
    MentHlth (0-30) or Income (1-8) would dominate the distance
    calculation over binary features otherwise.

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
