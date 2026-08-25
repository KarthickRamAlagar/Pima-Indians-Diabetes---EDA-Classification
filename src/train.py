"""
train.py
IN:  nothing directly -- pulls data via data_loader, cleans via preprocessing
OUT: trained model files (models/knn.pkl, models/nb.pkl), the fitted
     scaler (models/scaler.pkl), and a train/test split cache
     (artifacts/split.pkl) so evaluate.py tests on the exact same held-out
     data the models never saw during training.

Run with:  uv run python src/train.py
"""

import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE

from data_loader import load_raw_data
from preprocessing import clean_data, fit_scaler

RANDOM_STATE = 42
MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"
KNN_K_VALUES = [1, 3, 5, 7]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df = clean_data(load_raw_data())

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    joblib.dump((X_test, y_test), f"{ARTIFACTS_DIR}/split.pkl")

    scaler = fit_scaler(X_train, save_path=f"{MODELS_DIR}/scaler.pkl")
    X_train_scaled = scaler.transform(X_train)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print("Class balance after SMOTE:", dict(y_train_bal.value_counts()))

    # Train KNN across multiple k values so evaluate.py can compare them
    knn_models = {}
    for k in KNN_K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_bal, y_train_bal)
        knn_models[k] = model
        print(f"KNN (k={k}) trained.")
    joblib.dump(knn_models, f"{MODELS_DIR}/knn_multi.pkl")

    # Keep a single default (k=5) saved separately too, for anything that
    # only wants "the" KNN model rather than the full k comparison
    joblib.dump(knn_models[5], f"{MODELS_DIR}/knn.pkl")

    nb = GaussianNB()
    nb.fit(X_train_bal, y_train_bal)
    joblib.dump(nb, f"{MODELS_DIR}/nb.pkl")
    print("Naive Bayes trained and saved.")


if __name__ == "__main__":
    main()
