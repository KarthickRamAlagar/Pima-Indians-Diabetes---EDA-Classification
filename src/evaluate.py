"""
evaluate.py
IN:  saved models (models/knn.pkl, models/nb.pkl, models/scaler.pkl) and
     the held-out test split cached by train.py (artifacts/split.pkl)
OUT: printed classification reports + confusion matrices, and a
     metrics comparison table saved to artifacts/metrics.csv for the
     Streamlit app to display without retraining anything.

Run with:  uv run python src/evaluate.py   (after train.py has run once)
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

MODELS_DIR = "models"
ARTIFACTS_DIR = "artifacts"


def evaluate_model(model, X_test_scaled, y_test, name: str):
    pred = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


def main():
    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    knn = joblib.load(f"{MODELS_DIR}/knn.pkl")
    nb = joblib.load(f"{MODELS_DIR}/nb.pkl")
    X_test, y_test = joblib.load(f"{ARTIFACTS_DIR}/split.pkl")

    X_test_scaled = scaler.transform(X_test)

    results = [
        evaluate_model(knn, X_test_scaled, y_test, "KNN (k=5, uniform)"),
        evaluate_model(nb, X_test_scaled, y_test, "Naive Bayes"),
    ]

    knn_weighted_path = f"{MODELS_DIR}/knn_weighted.pkl"
    if os.path.exists(knn_weighted_path):
        knn_weighted = joblib.load(knn_weighted_path)
        results.append(evaluate_model(knn_weighted, X_test_scaled, y_test,
                                        "Weighted KNN (k=5, distance)"))

    results_df = pd.DataFrame(results).round(3)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    results_df.to_csv(f"{ARTIFACTS_DIR}/metrics.csv", index=False)
    print("\nSaved comparison table to artifacts/metrics.csv")
    print(results_df)

    # Uniform KNN across multiple k values
    knn_multi_path = f"{MODELS_DIR}/knn_multi.pkl"
    if os.path.exists(knn_multi_path):
        knn_models = joblib.load(knn_multi_path)
        k_results = []
        for k, model in sorted(knn_models.items()):
            row = evaluate_model(model, X_test_scaled, y_test, f"KNN (k={k})")
            row["k"] = k
            k_results.append(row)
        k_results_df = pd.DataFrame(k_results)[
            ["k", "Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        ].round(3)
        k_results_df.to_csv(f"{ARTIFACTS_DIR}/knn_k_metrics.csv", index=False)
        print("\nSaved KNN k-comparison table to artifacts/knn_k_metrics.csv")
        print(k_results_df)
    else:
        print("\nNo models/knn_multi.pkl found — rerun train.py to enable the "
              "KNN k-comparison (requires the updated train.py).")

    # Weighted KNN across multiple k values
    knn_weighted_multi_path = f"{MODELS_DIR}/knn_weighted_multi.pkl"
    if os.path.exists(knn_weighted_multi_path):
        knn_weighted_models = joblib.load(knn_weighted_multi_path)
        kw_results = []
        for k, model in sorted(knn_weighted_models.items()):
            row = evaluate_model(model, X_test_scaled, y_test, f"Weighted KNN (k={k})")
            row["k"] = k
            kw_results.append(row)
        kw_results_df = pd.DataFrame(kw_results)[
            ["k", "Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        ].round(3)
        kw_results_df.to_csv(f"{ARTIFACTS_DIR}/knn_weighted_k_metrics.csv", index=False)
        print("\nSaved Weighted KNN k-comparison table to "
              "artifacts/knn_weighted_k_metrics.csv")
        print(kw_results_df)
    else:
        print("\nNo models/knn_weighted_multi.pkl found — rerun train.py to "
              "enable the Weighted KNN k-comparison.")


if __name__ == "__main__":
    main()
