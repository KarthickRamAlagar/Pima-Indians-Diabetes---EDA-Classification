"""
generate_report.py
IN:  artifacts/metrics.csv and artifacts/knn_k_metrics.csv (from evaluate.py)
     plus the live dataset (via data_loader + preprocessing)
OUT: assignment1_pima_diabetes_report.pdf written to the project root,
     matching everything currently shown in the Streamlit dashboard
     (data quality, all 5 EDA plots, KNN vs Naive Bayes, KNN k-comparison,
     best-model comparison).

Run with:  uv run python src/generate_report.py
Then push the resulting PDF to your GitHub repo to update the hosted link
used by the dashboard's download button.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_raw_data
from preprocessing import clean_data
from visualize import (
    plot_boxplots, plot_histograms, plot_scatter,
    plot_correlation_heatmap, plot_class_distribution,
)
from report import generate_pdf

ARTIFACTS_DIR = "artifacts"
OUTPUT_PATH = "assignment1_pima_diabetes_report.pdf"


def build_knn_k_chart(knn_k_df: pd.DataFrame):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metric_cols:
        ax.plot(knn_k_df["k"], knn_k_df[metric], marker="o", label=metric)
    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("Score")
    ax.set_xticks(knn_k_df["k"])
    ax.set_title("KNN evaluation metrics vs k")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def build_best_compare(knn_k_df: pd.DataFrame, metrics_df: pd.DataFrame):
    best_knn_row = knn_k_df.loc[knn_k_df["F1-score"].idxmax()].copy()
    best_knn_row["Model"] = f"KNN (k={int(best_knn_row['k'])})"
    nb_row = metrics_df[metrics_df["Model"] == "Naive Bayes"].iloc[0]

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    compare_df = pd.DataFrame([
        {"Model": best_knn_row["Model"], **{m: best_knn_row[m] for m in metric_cols}},
        {"Model": "Naive Bayes", **{m: nb_row[m] for m in metric_cols}},
    ]).round(3)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(metric_cols))
    width = 0.35
    ax.bar([i - width / 2 for i in x], compare_df.iloc[0][metric_cols], width,
           label=compare_df.iloc[0]["Model"], color="#7F77DD")
    ax.bar([i + width / 2 for i in x], compare_df.iloc[1][metric_cols], width,
           label=compare_df.iloc[1]["Model"], color="#1D9E75")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metric_cols)
    ax.set_ylabel("Score")
    ax.set_title("Best-tuned KNN vs Naive Bayes")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return compare_df, fig


def main():
    df = clean_data(load_raw_data())
    quality_df = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "cardinality": df.nunique(),
    })

    metrics_path = f"{ARTIFACTS_DIR}/metrics.csv"
    knn_k_path = f"{ARTIFACTS_DIR}/knn_k_metrics.csv"
    if not os.path.exists(metrics_path) or not os.path.exists(knn_k_path):
        raise SystemExit(
            "Missing artifacts. Run `uv run python src/train.py` then "
            "`uv run python src/evaluate.py` first."
        )
    metrics_df = pd.read_csv(metrics_path)
    knn_k_df = pd.read_csv(knn_k_path)

    figures = {
        "Boxplots": plot_boxplots(df),
        "Histograms": plot_histograms(df),
        "Scatter: Glucose vs BMI": plot_scatter(df),
        "Correlation heatmap": plot_correlation_heatmap(df),
        "Class distribution": plot_class_distribution(df),
    }

    knn_k_fig = build_knn_k_chart(knn_k_df)
    best_compare_df, best_compare_fig = build_best_compare(knn_k_df, metrics_df)

    pdf_bytes = generate_pdf(
        df, quality_df, figures, metrics_df,
        knn_k_df=knn_k_df, knn_k_fig=knn_k_fig,
        best_compare_df=best_compare_df, best_compare_fig=best_compare_fig,
    )

    with open(OUTPUT_PATH, "wb") as f:
        f.write(pdf_bytes)
    print(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
