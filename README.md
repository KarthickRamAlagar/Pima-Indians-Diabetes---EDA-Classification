# Pima Diabetes ML — Assignment 1 (26DS601)


<p align="center">

**Exploratory Data Analysis & Machine Learning Classification on the Pima Indians Diabetes Dataset**

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/uv-6A5ACD?style=for-the-badge&logo=python&logoColor=white" alt="uv"/>
</p>
<p align="center">

👉 <a href="https://pima-indians-diabetes---eda-classification.streamlit.app/"><strong>Open Pima Diabetes ML Dashboard</strong></a>

</p>
---

# 📌 Overview

This project is **Assignment 1 (26DS601)** focused on Exploratory Data Analysis (EDA) and machine learning classification using the **Pima Indians Diabetes Dataset**.

The project implements a structured machine learning workflow covering:

```text
Data Loading
     ↓
Data Cleaning
     ↓
Missing Value Imputation
     ↓
Exploratory Data Analysis
     ↓
Feature Scaling
     ↓
Train / Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Interactive Streamlit Dashboard
```

The dataset contains:

```text
768 rows
9 columns
```

The target variable is:

```text
Outcome
```

The project evaluates two classification algorithms:

```text
K-Nearest Neighbors (KNN)
Naive Bayes
```

---

# 🛠 Technology Stack

<table width="100%">
  <tr>
    <th align="left">Layer</th>
    <th align="left">Technology</th>
  </tr>

  <tr>
    <td>Programming Language</td>
    <td>Python 3.11+</td>
  </tr>

  <tr>
    <td>Package Manager</td>
    <td>uv</td>
  </tr>

  <tr>
    <td>Development Environment</td>
    <td>VS Code</td>
  </tr>

  <tr>
    <td>Data Manipulation</td>
    <td>Pandas</td>
  </tr>

  <tr>
    <td>Numerical Computing</td>
    <td>NumPy</td>
  </tr>

  <tr>
    <td>Visualization</td>
    <td>Matplotlib</td>
  </tr>

  <tr>
    <td>Statistical Visualization</td>
    <td>Seaborn</td>
  </tr>

  <tr>
    <td>Machine Learning</td>
    <td>Scikit-learn</td>
  </tr>

  <tr>
    <td>Imbalanced Learning</td>
    <td>imbalanced-learn</td>
  </tr>

  <tr>
    <td>Model Persistence</td>
    <td>Joblib</td>
  </tr>

  <tr>
    <td>Interactive Dashboard</td>
    <td>Streamlit</td>
  </tr>

  <tr>
    <td>Report Generation</td>
    <td>ReportLab</td>
  </tr>
</table>

---

# 📊 Dataset

The project uses the **Pima Indians Diabetes Dataset**.

The analyzed dataset contains:

```text
Rows       : 768
Columns    : 9
Target     : Outcome
```

The features are:

```text
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age
Outcome
```

The report identifies the following data types and cardinalities:

| Feature                    | Data Type | Cardinality |
| -------------------------- | --------- | ----------: |
| Pregnancies                | int64     |          17 |
| Glucose                    | float64   |         135 |
| Blood Pressure             | float64   |          47 |
| Skin Thickness             | float64   |          50 |
| Insulin                    | float64   |         187 |
| BMI                        | float64   |         247 |
| Diabetes Pedigree Function | float64   |         517 |
| Age                        | int64     |          52 |
| Outcome                    | int64     |           2 |

---

# 🏗 Project Architecture

```text
                         PIMA DIABETES DATASET
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  data_loader.py  │
                         │   Data Loading   │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  preprocessing.py   │
                       │                     │
                       │ Zero → NaN          │
                       │ Median Imputation   │
                       │ Standard Scaling    │
                       └──────────┬──────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        ┌─────────────────┐               ┌─────────────────┐
        │  visualize.py   │               │    train.py     │
        │                 │               │                 │
        │ EDA             │               │ Train KNN       │
        │ Boxplots        │               │ Train NB        │
        │ Histograms      │               │ Save Models     │
        │ Scatter Plots   │               │ Save Scaler     │
        │ Correlation     │               │ Cache Split     │
        └────────┬────────┘               └────────┬────────┘
                 │                                 │
                 ▼                                 ▼
        ┌─────────────────┐               ┌─────────────────┐
        │    Figures      │               │   evaluate.py   │
        │ artifacts/      │               │                 │
        │ figures/        │               │ Metrics         │
        └─────────────────┘               │ Confusion Matrix│
                                          │ Classification  │
                                          │ Reports         │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │  artifacts/     │
                                          │  metrics.csv    │
                                          └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │     app.py      │
                                          │   Streamlit     │
                                          │    Dashboard    │
                                          └─────────────────┘
```

---

# 📁 Project Structure

```text
pima-diabetes-ml/
│
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualize.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py
│
├── models/
│   ├── knn.pkl
│   ├── nb.pkl
│   └── scaler.pkl
│
├── artifacts/
│   ├── split.pkl
│   ├── metrics.csv
│   └── figures/
│
└── data/
    └── ...
```

---

# 🚀 Setup — uv + VS Code

## 1. Clone Repository

```bash
git clone https://github.com/KarthickRamAlagar/Pima-Indians-Diabetes---EDA-Classification.git
```

```bash
cd Pima-Indians-Diabetes---EDA-Classification
```

---

## 2. Install Dependencies

From the project root:

```bash
uv sync
```

This installs the dependencies defined in `pyproject.toml` and creates the project's `.venv` environment.

---

## 3. Open in VS Code

Open the project folder in VS Code.

Select the `.venv` Python interpreter created by `uv` from the Python interpreter selector in VS Code.

The project requires:

```text
Python >= 3.11
```

---

# ▶️ Running the Project

## Train Models

```bash
uv run python src/train.py
```

The training pipeline:

```text
Load Data
    ↓
Clean Data
    ↓
Split Data
    ↓
Scale Features
    ↓
Train KNN
    ↓
Train Naive Bayes
    ↓
Save Models
```

Generated artifacts include:

```text
models/knn.pkl
models/nb.pkl
models/scaler.pkl
artifacts/split.pkl
```

---

## Evaluate Models

```bash
uv run python src/evaluate.py
```

The evaluation pipeline loads the saved models and cached test split.

It produces:

```text
Classification Reports
Confusion Matrices
artifacts/metrics.csv
```

---

## Launch Streamlit Dashboard

```bash
uv run streamlit run src/app.py
```

This launches the interactive Streamlit dashboard.

---

# 📦 Module Reference

| File                   | Input                                  | Output                                                                        |
| ---------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| `src/data_loader.py`   | URL or local CSV path                  | Raw `DataFrame`, named columns, unmodified values                             |
| `src/preprocessing.py` | Raw `DataFrame`                        | Cleaned `DataFrame`; fitted `StandardScaler` saved to `models/scaler.pkl`     |
| `src/visualize.py`     | Cleaned `DataFrame`                    | PNG figures in `artifacts/figures/` or in-memory `Figure` objects             |
| `src/train.py`         | Dataset                                | `models/knn.pkl`, `models/nb.pkl`, `models/scaler.pkl`, `artifacts/split.pkl` |
| `src/evaluate.py`      | Saved models + `artifacts/split.pkl`   | Classification reports, confusion matrices, `artifacts/metrics.csv`           |
| `src/app.py`           | Cleaned data + `artifacts/metrics.csv` | Interactive Streamlit dashboard                                               |

---

# 🔄 Data Preprocessing

The preprocessing pipeline handles missing or invalid values before model training.

The workflow is:

```text
Raw DataFrame
      │
      ▼
Identify Zero Values
      │
      ▼
Replace Zeros with NaN
      │
      ▼
Median Imputation
      │
      ▼
Clean DataFrame
      │
      ▼
StandardScaler
      │
      ▼
Scaled Features
```

The fitted scaler is saved as:

```text
models/scaler.pkl
```

This allows the same transformation to be reused during evaluation and application inference.

---

# 🔒 Data Leakage Prevention

A major design goal of this project is to keep training and evaluation separate.

The workflow uses:

```text
train.py
    │
    ├── Train models
    ├── Fit scaler
    └── Cache test split
             │
             ▼
        split.pkl
             │
             ▼
evaluate.py
    │
    └── Evaluate saved models
```

The test set is cached once and is not re-created during evaluation.

Similarly, the `StandardScaler` is fitted during training and reused later rather than being fitted again on evaluation data.

This prevents accidental information leakage between training and testing.

---

# 🔍 Exploratory Data Analysis

The project performs several EDA techniques.

## Boxplots

Used to examine feature distributions and potential outliers.

---

## Histograms

Used to understand the distribution of numerical features.

---

## Scatter Plot

The analysis includes:

```text
Glucose vs BMI
```

to visualize the relationship between glucose level and BMI.

---

## Correlation Heatmap

A correlation heatmap is used to understand relationships between numerical variables.

---

## Class Distribution

The target variable `Outcome` is analyzed to understand the distribution of the two classes.

---

# 🤖 Machine Learning Models

The project evaluates two classification algorithms.

## K-Nearest Neighbors

KNN predicts the class of an observation based on its nearest neighboring observations.

Multiple values of `k` are evaluated:

```text
k = 1
k = 3
k = 5
k = 7
```

---

## Naive Bayes

Naive Bayes is used as a probabilistic classification baseline and is compared against KNN.

---

# 📏 Evaluation Metrics

The models are evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

These metrics provide multiple perspectives on classification performance.

---

# 📈 Model Results

The initial model comparison reported:

| Model       | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ----------- | -------: | --------: | -----: | -------: | ------: |
| KNN         |    0.773 |     0.638 |  0.815 |    0.715 |   0.844 |
| Naive Bayes |    0.734 |     0.600 |  0.722 |    0.655 |   0.813 |

KNN achieved stronger reported results than Naive Bayes across the comparison metrics.

---

# ⚙️ KNN Hyperparameter Analysis

KNN was evaluated using multiple values of `k`.

|  k | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| -: | -------: | --------: | -----: | -------: | ------: |
|  1 |    0.799 |     0.717 |  0.704 |    0.710 |   0.777 |
|  3 |    0.792 |     0.657 |  0.852 |    0.742 |   0.851 |
|  5 |    0.773 |     0.638 |  0.815 |    0.715 |   0.844 |
|  7 |    0.773 |     0.627 |  0.870 |    0.729 |   0.857 |

---

# 🏆 Best Model

The final report compares the best-tuned KNN configuration against Naive Bayes:

| Model         |  Accuracy | Precision |    Recall |  F1-Score |   ROC-AUC |
| ------------- | --------: | --------: | --------: | --------: | --------: |
| **KNN (k=3)** | **0.792** | **0.657** | **0.852** | **0.742** | **0.851** |
| Naive Bayes   |     0.734 |     0.600 |     0.722 |     0.655 |     0.813 |

Based on the reported final comparison, **KNN (k=3)** is the selected best model.

---

# 📊 Visualization Outputs

When `visualize.py` is executed in script mode, generated figures are stored under:

```text
artifacts/figures/
```

The analysis includes:

```text
Boxplots
Histograms
Glucose vs BMI Scatter Plot
Correlation Heatmap
Class Distribution
```

The report contains these visualization categories.

---

# 🌐 Streamlit Dashboard

The project includes an interactive Streamlit application.

Run:

```bash
uv run streamlit run src/app.py
```

The application regenerates cleaned data live and uses the generated metrics for displaying model results.

The dashboard provides an interactive interface for presenting the project's EDA and machine learning results.

---

# 🗄 Generated Artifacts

The project produces the following important artifacts:

```text
models/
│
├── knn.pkl
├── nb.pkl
└── scaler.pkl

artifacts/
│
├── split.pkl
├── metrics.csv
└── figures/
```

### Models

```text
knn.pkl
nb.pkl
```

contain the trained KNN and Naive Bayes models.

### Scaler

```text
scaler.pkl
```

contains the fitted `StandardScaler`.

### Cached Split

```text
split.pkl
```

stores the held-out test split used for consistent evaluation.

### Metrics

```text
metrics.csv
```

contains the generated model evaluation metrics.

### Figures

```text
artifacts/figures/
```

contains generated visualization outputs when the visualization module is executed in script mode.

---

# 🧠 Why This Structure

The project is intentionally separated into independent modules.

### Training and Evaluation Separation

```text
train.py
    │
    ▼
Training
    │
    ▼
Saved Models
    │
    ▼
evaluate.py
    │
    ▼
Evaluation
```

Training and testing are separated so that the held-out test set is cached once and is not accidentally modified by training code.

---

### Single Scaler

The scaler is fitted once during training:

```text
Training Data
     │
     ▼
StandardScaler.fit()
     │
     ▼
scaler.pkl
     │
     ├──────────────► evaluate.py
     │
     └──────────────► app.py
```

This ensures that evaluation and application code reuse the fitted scaler rather than fitting a new scaler.

---

### Shared Visualization Module

`visualize.py` acts as a reusable visualization library.

```text
                 visualize.py
                /            \
               /              \
              ▼                ▼
       Script Pipeline     Streamlit App
```

This avoids maintaining separate implementations of the same plots.

---

# 📄 Generated EDA Report

The project includes an EDA and classification report containing:

```text
1. Data Quality Report
2. Visualizations
3. Model Results
4. KNN Performance Across k Values
5. Best Model Comparison
```

## The report documents the dataset characteristics, visual analysis, model performance, KNN hyperparameter results, and final model comparison.

# 📦 Python Dependencies

The project requires:

```text
pandas >= 2.2
numpy >= 1.26
matplotlib >= 3.8
seaborn >= 0.13
scikit-learn >= 1.4
imbalanced-learn >= 0.12
streamlit >= 1.36
joblib >= 1.4
reportlab >= 4.2
```

Python requirement:

```text
Python >= 3.11
```

Package management:

```text
uv
```

---

# 🔐 Git Configuration

The repository excludes local environments, Python cache files, and Streamlit local configuration.

```gitignore
.venv/
__pycache__/
*.pyc
.streamlit/
```

---

# 🎯 Project Workflow — Complete

```text
                    PIMA DIABETES DATASET
                              │
                              ▼
                     data_loader.py
                              │
                              ▼
                    preprocessing.py
                              │
                              ▼
                    Cleaned Dataset
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        visualize.py                       train.py
              │                               │
              ▼                               ▼
      EDA Visualizations              KNN + Naive Bayes
                                              │
                                              ▼
                                       Saved Models
                                              │
                                              ▼
                                        evaluate.py
                                              │
                                              ▼
                                      Model Evaluation
                                              │
                                              ▼
                                        metrics.csv
                                              │
                                              ▼
                                          app.py
                                              │
                                              ▼
                                   Streamlit Dashboard
```

---

# 📌 Assignment Information

```text
Course       : 26DS601
Assignment   : Assignment 1
Project      : Pima Diabetes ML
Task         : EDA & Classification
Dataset      : Pima Indians Diabetes Dataset
Language     : Python
Package Tool : uv
```

---

# 🎯 Key Results

```text
Dataset
768 × 9

Models
KNN
Naive Bayes

KNN Values
1
3
5
7

Best Reported Model
KNN (k=3)

Accuracy
0.792

Precision
0.657

Recall
0.852

F1-Score
0.742

ROC-AUC
0.851
```

The reported final comparison identifies KNN with `k=3` as the best-tuned KNN configuration against Naive Bayes.

---

# 👨‍💻 Author

**Karthick Ramalagar**

M.Tech Data Science

**Assignment 1 — 26DS601**

Built using:

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
imbalanced-learn
Streamlit
Joblib
ReportLab
uv
```
