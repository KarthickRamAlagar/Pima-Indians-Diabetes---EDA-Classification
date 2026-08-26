"""
visualize.py
IN:  cleaned DataFrame from preprocessing.clean_data()
OUT: PNG figures saved to artifacts/figures/, and (when called from
     Streamlit) matplotlib Figure objects returned directly for inline
     rendering instead of being saved to disk.

The CDC dataset has 21 features of two real kinds: 14 binary (0/1) and
7 ordinal/near-continuous (BMI, GenHlth, MentHlth, PhysHlth, Age,
Education, Income). Boxplots only make sense for the second group -
a boxplot of a 0/1 column is not informative - so ORDINAL_COLS is used
for boxplots/histograms, while the correlation heatmap covers all 21
features since correlation with a binary target is still meaningful for
binary predictors (point-biserial correlation).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")

BINARY_COLS = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
                "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
                "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
                "DiffWalk", "Sex"]
ORDINAL_COLS = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
ALL_FEATURE_COLS = BINARY_COLS + ORDINAL_COLS


def _save_or_return(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=110)
        plt.close(fig)
        return save_path
    return fig


def plot_boxplots(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.flatten(), ORDINAL_COLS):
        sns.boxplot(y=df[col], ax=ax, color="#7F77DD")
        ax.set_title(col)
    for ax in axes.flatten()[len(ORDINAL_COLS):]:
        ax.axis("off")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_histograms(df: pd.DataFrame, save_path: str = None):
    fig = df[ORDINAL_COLS].hist(bins=20, figsize=(16, 10),
                                  color="#1D9E75", edgecolor="black")[0][0].figure
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_scatter(df: pd.DataFrame, x="BMI", y="GenHlth", save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x=x, y=y, hue="Outcome", palette=["#378ADD", "#D85A30"],
                     alpha=0.3, ax=ax)
    ax.set_title(f"{x} vs {y} by diabetes outcome")
    return _save_or_return(fig, save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[ALL_FEATURE_COLS + ["Outcome"]].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (all 21 features + target)")
    return _save_or_return(fig, save_path)


def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x="Outcome", data=df, hue="Outcome", palette=["#378ADD", "#D85A30"],
                   legend=False, ax=ax)
    ax.set_title("Class distribution: Outcome")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No diabetes (0)", "Diabetes/prediabetes (1)"])
    return _save_or_return(fig, save_path)


def generate_all_figures(df: pd.DataFrame, out_dir: str = "artifacts/figures"):
    """Generate and save every required plot to out_dir. Returns dict of paths."""
    paths = {
        "boxplots": plot_boxplots(df, f"{out_dir}/boxplots.png"),
        "histograms": plot_histograms(df, f"{out_dir}/histograms.png"),
        "scatter": plot_scatter(df, save_path=f"{out_dir}/scatter.png"),
        "heatmap": plot_correlation_heatmap(df, f"{out_dir}/heatmap.png"),
        "class_distribution": plot_class_distribution(df, f"{out_dir}/class_distribution.png"),
    }
    return paths
