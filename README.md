# CDC Diabetes Health Indicators - Assignment 1 (26DS601)

Switched from the Pima Indians Diabetes dataset (768 rows, 9 columns) to
the **CDC Diabetes Health Indicators dataset** (253,680 rows, 21 features
+ binary target, UCI ML Repository id=891) to meet the faculty's
requirement of 10,000+ rows and 15+ features.

## Setup (uv + VS Code)

```bash
uv sync
uv run python src/train.py           # ~1-2 min: trains KNN (k=1,3,5,7) + Naive Bayes
uv run python src/evaluate.py        # evaluates on the held-out test set
uv run streamlit run src/app.py      # launches the interactive dashboard
```

First run fetches the dataset via `ucimlrepo` and caches it locally at
`data/cdc_diabetes.csv` - later runs need no network access.

## Key differences from the Pima version

- **No missing-value imputation needed.** The CDC dataset has zero missing
  values (already cleaned before publication). `preprocessing.clean_data()`
  is a pass-through - for Q1 task 6, the answer is "not needed" with a
  documented reason, not an imputation step.
- **ML training uses a 15,000-row stratified sample**, not the full
  253,680 rows. KNN's distance calculations don't scale well to that size
  for interactive use. EDA, the data quality report, and all
  visualizations still use the full dataset - only the KNN/Naive Bayes
  training step is sampled. This is a deliberate, documented scoping
  decision, not an oversight.
- **21 features split into two groups** for visualization purposes: 14
  binary (0/1) features and 7 ordinal/near-continuous features (BMI,
  GenHlth, MentHlth, PhysHlth, Age, Education, Income). Boxplots and
  histograms only cover the second group, since a boxplot of a 0/1 column
  isn't informative. The correlation heatmap covers all 21 features.

## Module reference (input / output)

| File | Input | Output |
|---|---|---|
| `src/data_loader.py` | UCI ML Repo fetch (id=891), cached locally after first run | Raw `DataFrame`, 21 features + `Outcome` |
| `src/preprocessing.py` | Raw `DataFrame` | Pass-through (no missing values); fitted `StandardScaler` saved to `models/scaler.pkl` |
| `src/visualize.py` | Cleaned `DataFrame` | PNG figures in `artifacts/figures/` or in-memory `Figure` objects |
| `src/train.py` | Nothing (pulls + cleans + samples data itself) | `models/knn_multi.pkl`, `models/knn.pkl`, `models/nb.pkl`, `models/scaler.pkl`, `artifacts/split.pkl` |
| `src/evaluate.py` | Saved models + `artifacts/split.pkl` | `artifacts/metrics.csv`, `artifacts/knn_k_metrics.csv` |
| `src/app.py` | Full dataset (EDA) + sampled-model artifacts | Streamlit dashboard |
| `src/generate_report.py` | Same artifacts as `app.py` | `assignment1_pima_diabetes_report.pdf` (filename kept for the existing GitHub link; content now reflects the CDC dataset) |
