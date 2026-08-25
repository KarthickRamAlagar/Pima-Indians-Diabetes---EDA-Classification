# Pima Diabetes ML — Assignment 1 (26DS601)

## Setup (uv + VS Code)

```bash
# from the project root
uv sync                              # installs everything from pyproject.toml
uv run python src/train.py           # trains + saves KNN and Naive Bayes
uv run python src/evaluate.py        # evaluates on the held-out test set
uv run streamlit run src/app.py      # launches the interactive dashboard
```

Open the folder in VS Code, select the `.venv` uv creates as your Python
interpreter (bottom-right of the VS Code window), and run any file directly
or via the integrated terminal with the commands above.

## Module reference (input / output)

| File | Input | Output |
|---|---|---|
| `src/data_loader.py` | URL (or local CSV path) | Raw `DataFrame`, named columns, unmodified values |
| `src/preprocessing.py` | Raw `DataFrame` | Cleaned `DataFrame` (zeros→NaN→median-imputed); fitted `StandardScaler` saved to `models/scaler.pkl` |
| `src/visualize.py` | Cleaned `DataFrame` | PNG figures in `artifacts/figures/` (script mode) or in-memory `Figure` objects (Streamlit mode) |
| `src/train.py` | Nothing (pulls + cleans data itself) | `models/knn.pkl`, `models/nb.pkl`, `models/scaler.pkl`, `artifacts/split.pkl` (cached test set) |
| `src/evaluate.py` | Saved models + `artifacts/split.pkl` | Console classification reports + confusion matrices; `artifacts/metrics.csv` |
| `src/app.py` | Cleaned data (regenerated live) + `artifacts/metrics.csv` | Interactive Streamlit web page (no file output) |

## Why this structure

- **Training and testing are separate** (`train.py` / `evaluate.py`) so the
  test set is cached once and never re-touched by training code — avoiding
  any accidental leakage between the two.
- **Scaling is fit once, in training, and reused** (`models/scaler.pkl`) —
  `evaluate.py` and `app.py` never re-fit a scaler, which would silently
  invalidate the results.
- **Visualization is a pure function library** (`visualize.py`) shared by
  both the script pipeline and the Streamlit app, so a plot never has two
  divergent implementations.
