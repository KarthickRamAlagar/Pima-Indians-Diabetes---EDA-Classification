"""
data_loader.py
IN:  nothing (fetches from the UCI ML Repository, first run only)
OUT: raw pandas DataFrame - 21 features + a renamed 'Outcome' target column
     (originally 'Diabetes_binary'; renamed here so the rest of the
     pipeline - train.py, evaluate.py, app.py - doesn't need to know the
     original CDC column name)

Dataset: CDC Diabetes Health Indicators (UCI ML Repo id=891), 253,680 rows,
21 features, binary target, zero missing values. See README for the
faculty requirement (10k+ rows, 15+ features) this dataset satisfies.

After the first successful fetch, the data is cached locally at
data/cdc_diabetes.csv so later runs (including your review demo) don't
depend on network access at all.
"""

import os
import pandas as pd

LOCAL_CACHE = "data/cdc_diabetes.csv"


def load_raw_data() -> pd.DataFrame:
    """Load the CDC Diabetes Health Indicators dataset, preferring a local cache.

    OUT: DataFrame with 21 feature columns + 'Outcome' (renamed from
         'Diabetes_binary'), exactly as published (no cleaning here)
    """
    if os.path.exists(LOCAL_CACHE):
        return pd.read_csv(LOCAL_CACHE)

    try:
        from ucimlrepo import fetch_ucirepo
        cdc = fetch_ucirepo(id=891)
        X = cdc.data.features
        y = cdc.data.targets.rename(columns={"Diabetes_binary": "Outcome"})
        df = pd.concat([X, y], axis=1)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch the dataset via ucimlrepo ({e}).\n"
            f"Run `uv add ucimlrepo` if it's not installed, or check your "
            f"network connection. Once fetched once, it's cached locally at "
            f"'{LOCAL_CACHE}' and won't need network again."
        ) from e

    os.makedirs(os.path.dirname(LOCAL_CACHE), exist_ok=True)
    df.to_csv(LOCAL_CACHE, index=False)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Loaded shape:", df.shape)
    print(df.head())
