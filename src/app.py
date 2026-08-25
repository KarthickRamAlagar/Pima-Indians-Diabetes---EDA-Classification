"""
app.py
IN:  cleaned dataset (regenerated live via data_loader + preprocessing) and
     artifacts/metrics.csv produced by evaluate.py (run train.py and
     evaluate.py at least once before launching this app)
OUT: an interactive Streamlit web page - no files written

Run with:  uv run streamlit run src/app.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from data_loader import load_raw_data
from preprocessing import clean_data
from visualize import (
    plot_boxplots, plot_histograms, plot_scatter,
    plot_correlation_heatmap, plot_class_distribution,
)

st.set_page_config(page_title="Pima Diabetes - EDA & Model Results", layout="wide")
st.title("Assignment 1: Pima Indians Diabetes - EDA & Classification")

@st.cache_data
def get_data():
    return clean_data(load_raw_data())

df = get_data()

quality_df = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "cardinality": df.nunique(),
})

metrics_path = "artifacts/metrics.csv"
metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else pd.DataFrame()

knn_k_path = "artifacts/knn_k_metrics.csv"
knn_k_df = pd.read_csv(knn_k_path) if os.path.exists(knn_k_path) else pd.DataFrame()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data quality", "Visualizations", "Model results", "KNN: k comparison",
     "Best model comparison"]
)

with tab1:
    st.subheader("Data quality report")
    st.dataframe(quality_df, width="stretch")
    st.caption("Missing values were already imputed by preprocessing.clean_data(); "
               "see README for the raw missing-value counts before imputation.")

with tab2:
    st.subheader("Boxplots")
    st.pyplot(plot_boxplots(df))

    st.subheader("Histograms")
    st.pyplot(plot_histograms(df))

    st.subheader("Scatter: Glucose vs BMI")
    st.pyplot(plot_scatter(df))

    st.subheader("Correlation heatmap")
    st.pyplot(plot_correlation_heatmap(df))

    st.subheader("Class distribution")
    st.pyplot(plot_class_distribution(df))

with tab3:
    st.subheader("KNN vs Naive Bayes")
    if not metrics_df.empty:
        st.dataframe(metrics_df, width="stretch")
    else:
        st.warning("Run `uv run python src/train.py` then "
                    "`uv run python src/evaluate.py` first to generate results.")

with tab4:
    st.subheader("KNN performance across k values")
    if not knn_k_df.empty:
        st.dataframe(knn_k_df, width="stretch")

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
        st.pyplot(fig)

        best_k_row = knn_k_df.loc[knn_k_df["F1-score"].idxmax()]
        st.caption(f"Best F1-score at k={int(best_k_row['k'])} "
                   f"(F1={best_k_row['F1-score']:.3f}).")
    else:
        st.warning("Run `uv run python src/train.py` then "
                    "`uv run python src/evaluate.py` first (with the updated "
                    "train.py) to generate the k-comparison results.")

with tab5:
    st.subheader("Best KNN (by k) vs Naive Bayes")
    if not knn_k_df.empty and not metrics_df.empty:
        best_knn_row = knn_k_df.loc[knn_k_df["F1-score"].idxmax()].copy()
        best_knn_row["Model"] = f"KNN (k={int(best_knn_row['k'])})"
        nb_row = metrics_df[metrics_df["Model"] == "Naive Bayes"].iloc[0]

        metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        compare_df = pd.DataFrame([
            {"Model": best_knn_row["Model"], **{m: best_knn_row[m] for m in metric_cols}},
            {"Model": "Naive Bayes", **{m: nb_row[m] for m in metric_cols}},
        ])
        st.dataframe(compare_df, width="stretch")

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
        st.pyplot(fig)

        winner = compare_df.loc[compare_df["F1-score"].idxmax(), "Model"]
        st.caption(f"Higher F1-score: {winner}.")
    else:
        st.warning("Run `uv run python src/train.py` then "
                    "`uv run python src/evaluate.py` first to generate results.")

# --- Floating round download button (links directly to the hosted PDF) ---
PDF_URL = ("https://raw.githubusercontent.com/KarthickRamAlagar/"
           "Pima-Indians-Diabetes---EDA-Classification/main/"
           "assignment1_pima_diabetes_report%20(1).pdf")

fab_html = f"""
<style>
html, body {{ margin: 0; padding: 0; background: transparent; }}
@keyframes fab-bounce {{
    0%, 50%, 100% {{ transform: translateY(0); }}
    25% {{ transform: translateY(-10px); }}
}}
.fab-wrap {{ display: flex; justify-content: flex-end; padding: 8px 16px; }}
#pdf-fab {{
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: #534AB7;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    box-shadow: 0 3px 10px rgba(0,0,0,0.3);
    animation: fab-bounce 2.4s ease-in-out infinite;
}}
#pdf-fab:hover {{ background: #3C3489; }}
#pdf-fab svg {{ width: 24px; height: 24px; }}
</style>
<div class="fab-wrap">
    <a id="pdf-fab" href="{PDF_URL}" download title="Download report (PDF)">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6 2h9l5 5v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z" fill="#fff"/>
            <path d="M15 2v5h5" fill="none" stroke="#534AB7" stroke-width="1.2"/>
            <text x="12" y="17" text-anchor="middle" font-family="Helvetica, Arial, sans-serif"
                  font-size="6.5" font-weight="700" fill="#534AB7">PDF</text>
        </svg>
    </a>
</div>
<script>
document.getElementById('pdf-fab').addEventListener('click', function() {{
    try {{
        var ctx = new (window.AudioContext || window.webkitAudioContext)();
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.value = 1046.5;
        gain.gain.setValueAtTime(0.25, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
        osc.connect(gain); gain.connect(ctx.destination);
        osc.start(); osc.stop(ctx.currentTime + 0.3);
    }} catch (e) {{}}
}});
</script>
"""
st.iframe(fab_html, height=80)
