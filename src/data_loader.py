"""
data_loader.py
IN:  nothing (fetches from a public URL) -- optionally a local CSV path
OUT: raw pandas DataFrame, unmodified except for column naming
"""

import pandas as pd

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

DEFAULT_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)


def load_raw_data(source: str = DEFAULT_URL) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset from a URL or local CSV path.

    IN:  source - URL or filepath to the raw CSV (no header row)
    OUT: DataFrame with named columns, exactly as published (no cleaning yet)
    """
    df = pd.read_csv(source, names=COLUMNS)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Loaded shape:", df.shape)
    print(df.head())
