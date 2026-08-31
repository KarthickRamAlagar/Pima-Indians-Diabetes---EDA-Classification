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

st.set_page_config(page_title="CDC Diabetes Health Indicators - EDA & Model Results", layout="wide")
st.title("Assignment 1: CDC Diabetes Health Indicators - EDA & Classification")
st.markdown("## **SDG 3: Good Health and Well-being**")
st.caption("253,680 rows, 21 features (CDC BRFSS 2015, via UCI ML Repository id=891). "
           "EDA uses the full dataset; KNN/Naive Bayes are trained on a 15,000-row "
           "stratified sample for interactive performance (see README).")

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

knn_weighted_k_path = "artifacts/knn_weighted_k_metrics.csv"
knn_weighted_k_df = pd.read_csv(knn_weighted_k_path) if os.path.exists(knn_weighted_k_path) else pd.DataFrame()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data quality", "Visualizations", "Model results", "KNN: k comparison",
     "Best model comparison"]
)
# Note: tab4 now includes a weighting-scheme toggle (uniform vs distance-weighted KNN)

with tab1:
    st.subheader("Data quality report")
    st.dataframe(quality_df, width="stretch")
    st.caption("Missing values were already imputed by preprocessing.clean_data(); "
               "see README for the raw missing-value counts before imputation.")

import io

def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

@st.cache_resource
def cached_boxplots():
    return _fig_to_png_bytes(plot_boxplots(df))

@st.cache_resource
def cached_histograms():
    return _fig_to_png_bytes(plot_histograms(df))

@st.cache_resource
def cached_scatter():
    return _fig_to_png_bytes(plot_scatter(df))

@st.cache_resource
def cached_heatmap():
    return _fig_to_png_bytes(plot_correlation_heatmap(df))

@st.cache_resource
def cached_class_dist():
    return _fig_to_png_bytes(plot_class_distribution(df))

with tab2:
    st.subheader("Boxplots")
    st.image(cached_boxplots())

    st.subheader("Histograms")
    st.image(cached_histograms())

    st.subheader("Scatter: BMI vs GenHlth")
    st.image(cached_scatter())

    st.subheader("Correlation heatmap")
    st.image(cached_heatmap())

    st.subheader("Class distribution")
    st.image(cached_class_dist())

with tab3:
    st.subheader("KNN vs Weighted KNN vs Naive Bayes")
    if not metrics_df.empty:
        st.dataframe(metrics_df, width="stretch")
    else:
        st.warning("Run `uv run python src/train.py` then "
                    "`uv run python src/evaluate.py` first to generate results.")

with tab4:
    st.subheader("KNN performance across k values")
    weighting_choice = st.radio(
        "Weighting scheme",
        ["Uniform (all neighbors count equally)", "Distance-weighted (closer neighbors count more)"],
        horizontal=True,
    )
    active_df = knn_k_df if weighting_choice.startswith("Uniform") else knn_weighted_k_df
    chart_label = "Uniform KNN" if weighting_choice.startswith("Uniform") else "Weighted KNN"

    if not active_df.empty:
        st.dataframe(active_df, width="stretch")

        metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in metric_cols:
            ax.plot(active_df["k"], active_df[metric], marker="o", label=metric)
        ax.set_xlabel("k (number of neighbors)")
        ax.set_ylabel("Score")
        ax.set_xticks(active_df["k"])
        ax.set_title(f"{chart_label} evaluation metrics vs k")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        best_k_row = active_df.loc[active_df["F1-score"].idxmax()]
        st.caption(f"{chart_label}: best F1-score at k={int(best_k_row['k'])} "
                   f"(F1={best_k_row['F1-score']:.3f}).")
    else:
        st.warning("Run `uv run python src/train.py` then "
                    "`uv run python src/evaluate.py` first (with the updated "
                    "train.py) to generate the k-comparison results.")

with tab5:
    st.subheader("Best Uniform KNN vs Best Weighted KNN vs Naive Bayes")
    if not knn_k_df.empty and not metrics_df.empty:
        metric_cols = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

        best_knn_row = knn_k_df.loc[knn_k_df["F1-score"].idxmax()].copy()
        best_knn_row["Model"] = f"KNN (k={int(best_knn_row['k'])}, uniform)"
        nb_row = metrics_df[metrics_df["Model"] == "Naive Bayes"].iloc[0]

        rows = [
            {"Model": best_knn_row["Model"], **{m: best_knn_row[m] for m in metric_cols}},
            {"Model": "Naive Bayes", **{m: nb_row[m] for m in metric_cols}},
        ]

        if not knn_weighted_k_df.empty:
            best_wknn_row = knn_weighted_k_df.loc[knn_weighted_k_df["F1-score"].idxmax()].copy()
            best_wknn_row["Model"] = f"Weighted KNN (k={int(best_wknn_row['k'])})"
            rows.insert(1, {"Model": best_wknn_row["Model"],
                             **{m: best_wknn_row[m] for m in metric_cols}})

        compare_df = pd.DataFrame(rows)
        st.dataframe(compare_df, width="stretch")

        n_models = len(compare_df)
        colors = ["#7F77DD", "#E0A458", "#1D9E75"][:n_models]
        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(metric_cols))
        bar_width = 0.8 / n_models
        for i in range(n_models):
            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar([xi + offset for xi in x], compare_df.iloc[i][metric_cols],
                   bar_width, label=compare_df.iloc[i]["Model"], color=colors[i])
        ax.set_xticks(list(x))
        ax.set_xticklabels(metric_cols)
        ax.set_ylabel("Score")
        ax.set_title("Best-tuned KNN variants vs Naive Bayes")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close(fig)

        winner = compare_df.loc[compare_df["F1-score"].idxmax(), "Model"]
        st.caption(f"Higher F1-score: {winner}.")
    else:
        st.warning("Run `uv run python src/train.py` then "
                    "`uv run python src/evaluate.py` first to generate results.")

# --- Floating round download button (links directly to the hosted PDF) ---
PDF_URL = ("https://raw.githubusercontent.com/KarthickRamAlagar/"
           "Pima-Indians-Diabetes---EDA-Classification/main/"
           "CDC%20Diabetes%20Health%20Indicator%20-%2026034.pdf")

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
    <a id="pdf-fab" href="{PDF_URL}" download title="Download report (PDF)"
       onclick="
            try {{
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var playTone = function() {{
                    var osc = ctx.createOscillator();
                    var gain = ctx.createGain();
                    osc.type = 'sine';
                    osc.frequency.value = 1046.5;
                    gain.gain.setValueAtTime(0.25, ctx.currentTime);
                    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
                    osc.connect(gain); gain.connect(ctx.destination);
                    osc.start(); osc.stop(ctx.currentTime + 0.3);
                }};
                if (ctx.state === 'suspended') {{
                    ctx.resume().then(playTone);
                }} else {{
                    playTone();
                }}
            }} catch (e) {{}}
       ">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M6 2h9l5 5v15a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1z" fill="#fff"/>
            <path d="M15 2v5h5" fill="none" stroke="#534AB7" stroke-width="1.2"/>
            <text x="12" y="17" text-anchor="middle" font-family="Helvetica, Arial, sans-serif"
                  font-size="6.5" font-weight="700" fill="#534AB7">PDF</text>
        </svg>
    </a>
</div>
"""
st.markdown(fab_html, unsafe_allow_html=True)
