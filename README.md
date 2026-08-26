# 📘 CDC DIABETES HEALTH INDICATORS — EDA & CLASSIFICATION

**Exploratory Data Analysis and Machine Learning Classification on the CDC Diabetes Health Indicators Dataset using KNN and Naive Bayes.**
 <p align="center">
</p>

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>

<img src="https://img.shields.io/badge/Pandas-2.2+-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>

<img src="https://img.shields.io/badge/NumPy-1.26+-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>

<img src="https://img.shields.io/badge/Matplotlib-3.8+-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib"/>

<img src="https://img.shields.io/badge/Seaborn-0.13+-4C72B0?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn"/>

<img src="https://img.shields.io/badge/Scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>

<img src="https://img.shields.io/badge/Imbalanced--learn-0.12+-6A5ACD?style=for-the-badge&logo=python&logoColor=white" alt="Imbalanced-learn"/>

<img src="https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>

<img src="https://img.shields.io/badge/Joblib-1.4+-4B8BBE?style=for-the-badge&logo=python&logoColor=white" alt="Joblib"/>

<img src="https://img.shields.io/badge/ReportLab-4.2+-8B4513?style=for-the-badge&logo=python&logoColor=white" alt="ReportLab"/>

<img src="https://img.shields.io/badge/uv-Package_Manager-6A5ACD?style=for-the-badge&logo=python&logoColor=white" alt="uv"/>

<img src="https://img.shields.io/badge/VS_Code-Development_Environment-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VS Code"/>

<img src="https://img.shields.io/badge/Git-GitHub-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git"/>

</p>

<p align="center">
<a href="https://cdcdiabetes.streamlit.app/">
  <strong>🌐 Visit Live Streamlit Dashboard</strong>
</a>
</p>

## 📌 Overview

This project is **Assignment 1 (26DS601)** focused on Exploratory Data Analysis (EDA) and Machine Learning Classification using the **CDC Diabetes Health Indicators Dataset**.

The dataset is based on **CDC BRFSS 2015** data and contains:

```bash
253,680 Rows
22 Columns
````

The project performs:

```bash
Data Loading
Data Cleaning
Data Preprocessing
Missing Value Handling
Exploratory Data Analysis
Statistical Analysis
Feature Analysis
Data Visualization
KNN Classification
Naive Bayes Classification
Model Evaluation
KNN Hyperparameter Analysis
Model Comparison
Streamlit Dashboard
EDA & Classification Report
```

For machine learning experiments, a **stratified 15,000-row sample** was used.

---
## 🎓 Assignment Information

```bash
Course      : 26DS601
Assignment  : Assignment 1
Project     : CDC Diabetes Health Indicators
Task        : EDA & Classification
Language    : Python
Package Tool: uv
Dashboard   : Streamlit
Name        : KARTHIKEYAN R
Role Number : BL.SC.P2DSC26034
```


## 🛠 Technology Stack

| Layer                     | Technology       |
| ------------------------- | ---------------- |
| Programming Language      | Python 3.11+     |
| Package Manager           | uv               |
| Data Processing           | Pandas           |
| Numerical Computing       | NumPy            |
| Visualization             | Matplotlib       |
| Statistical Visualization | Seaborn          |
| Machine Learning          | Scikit-learn     |
| Imbalanced Learning       | imbalanced-learn |
| Model Persistence         | Joblib           |
| Dashboard                 | Streamlit        |
| Report Generation         | ReportLab        |
| Development Environment   | VS Code          |

---

## 📊 Dataset

<b>Dataset:</b>

```bash
CDC Diabetes Health Indicators
```

<b>Source:</b>

```bash
CDC BRFSS 2015
UCI Machine Learning Repository
Repository ID: 891
```

<b>Dataset Size:</b>

```bash
Rows    : 253,680
Columns : 22
```

<b>Target Variable:</b>

```bash
Outcome
```

---

## 🧬 Dataset Features

The dataset contains the following 22 features:

```bash
HighBP
HighChol
CholCheck
BMI
Smoker
Stroke
HeartDiseaseorAttack
PhysActivity
Fruits
Veggies
HvyAlcoholConsump
AnyHealthcare
NoDocbcCost
GenHlth
MentHlth
PhysHlth
DiffWalk
Sex
Age
Education
Income
Outcome
```

---

## 🏗 System Architecture

<p align="center">

```text
                    CDC DIABETES DATASET
                            │
                            ▼
                    DATA LOADING
                            │
                            ▼
                  DATA PREPROCESSING
                            │
                            ▼
                    DATA CLEANING
                            │
                            ▼
                EXPLORATORY DATA ANALYSIS
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
        VISUALIZATION              FEATURE PREPARATION
              │                           │
              │                           ▼
              │                    TRAIN / TEST SPLIT
              │                           │
              │              ┌────────────┴────────────┐
              │              │                         │
              │              ▼                         ▼
              │             KNN                 NAIVE BAYES
              │              │                         │
              │              └────────────┬────────────┘
              │                           │
              │                           ▼
              │                    MODEL EVALUATION
              │                           │
              │                           ▼
              │                    MODEL COMPARISON
              │                           │
              └───────────────────────────┤
                                          ▼
                                STREAMLIT DASHBOARD
```

</p>

---

## 📦 Project Modules

### 📥 Data Loading Module

<b>`src/data_loader.py`</b>

Responsible for:

```bash
Load CDC Diabetes Dataset
Read CSV Data
Validate Dataset
Prepare Data for Processing
```

---

### 🧹 Data Preprocessing Module

<b>`src/preprocessing.py`</b>

Responsible for:

```bash
Data Cleaning
Missing Value Handling
Median Imputation
Feature Preparation
Train/Test Preparation
Feature Scaling
```

---

### 📊 Visualization Module

<b>`src/visualize.py`</b>

Responsible for:

```bash
Boxplots
Histograms
Scatter Plots
Correlation Heatmap
Class Distribution
Feature Analysis
EDA Visualizations
```

---

### 🤖 Model Training Module

<b>`src/train.py`</b>

Responsible for:

```bash
KNN Training
Naive Bayes Training
KNN Hyperparameter Testing
Model Evaluation
Model Persistence
Metric Generation
```

---

### 📄 Report Module

<b>`src/report.py`</b>

Responsible for generating the EDA and classification report containing:

```bash
Data Quality Analysis
EDA Visualizations
Model Results
KNN Comparison
Naive Bayes Results
Final Model Comparison
```

---

### 🌐 Streamlit Application

<b>`src/app.py`</b>

Provides an interactive dashboard for:

```bash
Dataset Overview
EDA
Data Visualization
Model Performance
KNN Analysis
Naive Bayes Analysis
Classification Metrics
Model Comparison
```

---

## 🔄 Machine Learning Workflow

```text
Raw CDC Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Missing Value Handling
       │
       ▼
Stratified Sampling
       │
       ▼
Feature Preparation
       │
       ▼
Train / Test Split
       │
       ▼
Feature Scaling
       │
       ├──────────────────┐
       ▼                  ▼
      KNN            Naive Bayes
       │                  │
       └────────┬─────────┘
                ▼
         Model Evaluation
                │
                ▼
       Accuracy / Precision
       Recall / F1 / ROC-AUC
                │
                ▼
         Model Comparison
                │
                ▼
       Streamlit Dashboard
```

---

## 🔍 Exploratory Data Analysis

The project performs multiple EDA techniques.

<b>Distribution Analysis</b>

```bash
Histograms
Boxplots
Class Distribution
```

<b>Relationship Analysis</b>

```bash
Scatter Plots
Feature Relationships
Correlation Analysis
```

<b>Correlation Analysis</b>

```bash
Correlation Matrix
Heatmap Visualization
```

<b>Feature Analysis</b>

```bash
BMI
HighBP
HighChol
GenHlth
Age
Income
Education
```

---

## 🧹 Data Preprocessing

The preprocessing pipeline includes:

```bash
Data Cleaning
       ↓
Missing / Invalid Value Identification
       ↓
Class-Conditional Median Imputation
       ↓
Feature Preparation
       ↓
Feature Scaling
       ↓
Model Ready Dataset
```

The report uses a **stratified 15,000-row sample** for the machine learning experiments.

---

## 🤖 Machine Learning Models

### 📍 K-Nearest Neighbors

KNN classification is implemented and evaluated using different values of `k`.

```bash
k = 1
k = 3
k = 5
k = 7
```

The project evaluates the effect of different neighborhood sizes on classification performance.

---

### 🧠 Naive Bayes

Naive Bayes is implemented as a probabilistic classification model and compared against KNN.

The final comparison evaluates:

```bash
KNN (k=5)
Naive Bayes
```

---

## 📈 Model Evaluation

The following evaluation metrics are used:

```bash
Accuracy
Precision
Recall
F1-Score
ROC-AUC
```

These metrics provide a broader evaluation of classification performance rather than relying only on accuracy.

---

## 🏆 Model Results

### Final Model Comparison

| Model           | Accuracy  | Precision | Recall    | F1-Score  | ROC-AUC   |
| --------------- | --------- | --------- | --------- | --------- | --------- |
| KNN (k=5)       | 0.704     | 0.261     | 0.612     | 0.366     | 0.709     |
| **Naive Bayes** | **0.712** | **0.290** | **0.734** | **0.416** | **0.781** |

### 🥇 Best Reported Model

```bash
Naive Bayes
```

<b>Accuracy:</b>

```bash
0.712
```

<b>Precision:</b>

```bash
0.290
```

<b>Recall:</b>

```bash
0.734
```

<b>F1-Score:</b>

```bash
0.416
```

<b>ROC-AUC:</b>

```bash
0.781
```

Based on the reported final comparison, **Naive Bayes achieved the stronger overall performance**, particularly in Recall, F1-Score, and ROC-AUC.

---

## ⚙️ KNN Hyperparameter Analysis

The project evaluates KNN using:

```bash
k = 1
k = 3
k = 5
k = 7
```

| K | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| - | -------- | --------- | ------ | -------- | ------- |
| 1 | 0.772    | 0.281     | 0.409  | 0.333    | 0.620   |
| 3 | 0.726    | 0.261     | 0.526  | 0.349    | 0.684   |
| 5 | 0.704    | 0.261     | 0.612  | 0.366    | 0.709   |
| 7 | 0.683    | 0.253     | 0.651  | 0.364    | 0.725   |

---

## 💾 Model Artifacts

The trained models and generated artifacts are stored in:

```bash
models/
│
├── knn.pkl
├── knn_multi.pkl
├── nb.pkl
└── scaler.pkl
```

Generated evaluation artifacts:

```bash
artifacts/
│
├── split.pkl
├── metrics.csv
└── knn_k_metrics.csv
```

---

## 📁 Project Structure

```bash
pima-diabetes-ml/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── data/
│   └── cdc_diabetes.csv
│
├── models/
│   ├── knn.pkl
│   ├── knn_multi.pkl
│   ├── nb.pkl
│   └── scaler.pkl
│
├── artifacts/
│   ├── split.pkl
│   ├── metrics.csv
│   └── knn_k_metrics.csv
│
└── src/
    ├── data_loader.py
    ├── preprocessing.py
    ├── visualize.py
    ├── train.py
    ├── report.py
    └── app.py
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/KarthickRamAlagar/Pima-Indians-Diabetes---EDA-Classification.git
```

### Navigate to Project

```bash
cd Pima-Indians-Diabetes---EDA-Classification
```

### Install Dependencies

This project uses **uv** as the Python package manager.

```bash
uv sync
```

---

## ▶️ Run Machine Learning Pipeline

```bash
uv run python src/train.py
```

This runs the model training workflow and generates the required model and metric artifacts.

---

## 📊 Run Streamlit Dashboard

```bash
uv run streamlit run src/app.py
```

### 🌐 Live Dashboard

<p align="center">

<a href="https://cdcdiabetes.streamlit.app/">

🚀 <strong>Visit CDC Diabetes Health Indicators Dashboard</strong>

</a>

</p>

---

## 📄 Generate Report

```bash
uv run python src/report.py
```

The generated report is:

```bash
CDC Diabetes Health Indicator - 26034.pdf
```

---

## 📦 Dependencies

The project uses:

```bash
pandas>=2.2
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
imbalanced-learn>=0.12
streamlit>=1.36
joblib>=1.4
reportlab>=4.2
```

Python version:

```bash
Python >= 3.11
```

---

## 🧩 uv Package Management

Initialize the project:

```bash
uv init
```

Install dependencies:

```bash
uv sync
```

Add a new dependency:

```bash
uv add <package-name>
```

Run a Python file:

```bash
uv run python src/train.py
```

Run Streamlit:

```bash
uv run streamlit run src/app.py
```
---

## 🎯 Key Highlights

```bash
CDC BRFSS 2015 Dataset
253,680 Rows
22 Features
15,000 Stratified ML Sample

Exploratory Data Analysis
Data Preprocessing
Feature Scaling
KNN Classification
Naive Bayes Classification
KNN Hyperparameter Analysis
Model Evaluation
Streamlit Dashboard
Automated Report Generation
```

---

## 🔮 Future Improvements

```bash
Hyperparameter Optimization
Cross-Validation
Additional Classification Models
Random Forest
Logistic Regression
Gradient Boosting
XGBoost
Feature Importance Analysis
SHAP Explainability
Advanced Imbalanced-Learning Techniques
Model Comparison Dashboard
Prediction Interface
Model Monitoring
Cloud Deployment
CI/CD Pipeline
```

---

## 📄 Project Report

The complete EDA and Machine Learning report is included as:

<p align="center">

<a href="https://github.com/KarthickRamAlagar/Pima-Indians-Diabetes---EDA-Classification/blob/main/CDC%20Diabetes%20Health%20Indicator%20-%2026034.pdf">
  <strong>🫱 Download EDA & Model Report</strong>
</a>

</p>


The report covers:

```bash
Dataset Description
Data Quality Analysis
Exploratory Data Analysis
Feature Analysis
KNN Classification
Naive Bayes Classification
KNN Hyperparameter Analysis
Model Evaluation
Final Model Comparison
```
---

## 👨‍💻 Author

**Karthick Ramalagar**

M.Tech Data Science

Assignment 1 — **26DS601**

Built using:

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* imbalanced-learn
* Streamlit
* Joblib
* ReportLab
* uv

---

