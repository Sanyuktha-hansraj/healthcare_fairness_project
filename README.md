# 🏥 Healthcare Fairness ML: Bias Detection in Skin Lesion Classification

A Responsible AI and Healthcare Machine Learning project focused on identifying demographic disparities in dermatology datasets and developing fairness-aware evaluation workflows for skin lesion classification.

---

## 📋 Table of Contents

* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Key Features](#key-features)
* [Dataset](#dataset)
* [Key Findings](#key-findings)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Fairness Metrics](#fairness-metrics)
* [Dashboard](#dashboard)
* [Dependencies](#dependencies)
* [Future Work](#future-work)
* [References](#references)
* [Author](#author)

---

# 🎯 Overview

Healthcare AI systems can unintentionally perform differently across demographic groups, potentially leading to unequal healthcare outcomes.

This project investigates fairness in dermatological AI using the Diverse Dermatology Images (DDI) dataset. It focuses on analyzing differences across Fitzpatrick skin tone groups and building a configurable framework for fairness-aware model evaluation.

The project combines:

* Healthcare Data Science
* Responsible AI
* Fairness in Machine Learning
* Medical Image Analytics
* Exploratory Data Analysis
* Interactive Dashboard Development

The long-term goal is to build fairness-aware machine learning pipelines that support equitable healthcare AI development.

---

# ⚠️ Problem Statement

Machine learning models used in healthcare may perform differently across demographic groups.

In dermatology, this issue is particularly important because:

* Darker skin tones are often underrepresented in medical datasets.
* Performance disparities may contribute to delayed or missed diagnosis.
* Healthcare AI systems should be evaluated for fairness before clinical deployment.

This project aims to identify dataset-level disparities and prepare a fairness-aware evaluation framework for future model development.

---

# ✨ Key Features

## Fairness-Focused Analysis

The project includes fairness metric configuration, exploratory bias analysis, and a framework for future mitigation strategies.

Configured metrics include:

* Demographic Parity
* Equal Opportunity
* Equalized Odds
* Calibration Within Groups

## Exploratory Data Analysis

* Demographic breakdowns
* Skin tone distribution analysis
* Malignancy rate analysis
* Outcome disparity assessment

## Interactive Dashboard

Built using Streamlit for:

* Dataset exploration
* Cohort analysis
* Fairness observations
* Statistical summaries

## Configurable Pipeline

YAML-based configuration enables reproducible experiments and fairness evaluation settings.

## Feature Engineering

Automated extraction and preparation of metadata-derived features for future modeling tasks.

---

# 📊 Dataset

## Diverse Dermatology Images (DDI)

The DDI dataset was created to improve representation of diverse skin tones in dermatology research.

### Dataset Summary

| Attribute         | Value                    |
| ----------------- | ------------------------ |
| Total Images      | 656                      |
| Malignant Lesions | 171                      |
| Benign Lesions    | 485                      |
| Skin Tone Groups  | 3 Fitzpatrick Categories |

### Skin Tone Distribution

| Fitzpatrick Group  | Images |
| ------------------ | ------ |
| Fitzpatrick I-II   | 208    |
| Fitzpatrick III-IV | 241    |
| Fitzpatrick V-VI   | 207    |

**Note:** Please ensure proper licensing and citation when using the DDI dataset.

---

# 📈 Key Findings

## Dataset Characteristics

* Total images analyzed: 656
* Malignant lesions: 171
* Benign lesions: 485
* All demographic groups contain sufficient samples for cohort-level analysis
* 12 engineered features prepared for future model development

## Outcome Distribution

| Skin Tone Group    | Total Images | Malignant Cases | Malignancy Rate |
| ------------------ | ------------ | --------------- | --------------- |
| Fitzpatrick I-II   | 208          | 49              | 23.56%          |
| Fitzpatrick III-IV | 241          | 74              | 30.71%          |
| Fitzpatrick V-VI   | 207          | 48              | 23.19%          |

## Fairness Observation

The Fitzpatrick III-IV cohort demonstrates a higher malignancy prevalence compared with the other groups.

While this does not indicate algorithmic bias by itself, it highlights the importance of fairness-aware evaluation during model development.

**Outcome disparity observed across groups: 7.52 percentage points**

---

# 📁 Project Structure

```text
fairness_project/
├── config/
│   ├── fairness.yml
│   └── features.yml
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       └── 01_eda_cohort.py
│
├── notebooks/
│   └── 02_eda.ipynb
│
├── reports/
│   ├── log.md
│   ├── week2_log.md
│   └── figures/
│
├── src/
│   ├── data_loader.py
│   ├── fairness_metrics.py
│   ├── features.py
│   └── __init__.py
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/Sanyuktha-hansraj/healthcare_fairness_project.git
cd healthcare_fairness_project
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 💻 Usage

## Run Exploratory Data Analysis

```bash
jupyter notebook notebooks/02_eda.ipynb
```

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Example Dataset Loading

```python
from src.data_loader import DDI_DataLoader

loader = DDI_DataLoader(root_dir="./DDI")
df = loader.get_metadata()
```

## Example Fairness Evaluation

```python
from src.fairness_metrics import FairnessEvaluator

evaluator = FairnessEvaluator(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_prob,
    sensitive_attr=skin_tone_groups
)

dpd = evaluator.demographic_parity_difference()
```

---

# 📐 Fairness Metrics

## Demographic Parity

Measures whether positive prediction rates remain similar across demographic groups.

**Threshold:** < 0.1

## Equal Opportunity

Measures whether true positive rates remain similar across demographic groups.

**Threshold:** < 0.1

## Equalized Odds

Measures whether both true positive rates and false positive rates remain similar across groups.

**Threshold:** < 0.1

## Calibration Within Groups

Measures whether predicted probabilities correspond to actual outcomes within each demographic group.

**Threshold:** < 0.05

---

# 📊 Dashboard

## Current Dashboard Features

### Main Dashboard

* Project overview
* Dataset summary statistics
* Navigation interface

### Cohort Analysis Page

Implemented in:

```text
dashboard/pages/01_eda_cohort.py
```

Features:

* Dataset overview metrics
* Skin tone distribution visualization
* Malignancy rates by demographic group
* Outcome disparity analysis
* Statistical summaries
* Initial fairness observations

## Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📸 Dashboard Preview

Add screenshots here after exporting figures from Streamlit.

### Dataset Overview

![Dataset Overview](README_assets/dataset_overview.png)

### Skin Tone Distribution

![Skin Tone Distribution](README_assets/skin_tone_distribution.png)

### Malignancy Rates by Group

![Malignancy Distribution](README_assets/malignancy_distribution.png)

---

# 📚 Dependencies

| Package      | Purpose                   |
| ------------ | ------------------------- |
| Pandas       | Data processing           |
| NumPy        | Numerical computing       |
| PyTorch      | Deep learning             |
| Scikit-learn | Machine learning          |
| Fairlearn    | Fairness evaluation       |
| Streamlit    | Dashboard development     |
| Matplotlib   | Visualization             |
| Seaborn      | Statistical visualization |

---

# 🔬 Future Work

* Baseline skin lesion classification models
* Fairness-aware model evaluation
* Bias mitigation techniques
* Reweighting approaches
* Threshold optimization
* Calibration analysis
* Model comparison dashboard
* Clinical fairness benchmarking

---

# 📖 References

## Dataset

* Diverse Dermatology Images (DDI) Dataset

## Libraries

* Fairlearn
* Streamlit
* PyTorch
* Scikit-learn
* Pandas
* NumPy

---

# 👩‍💻 Author

**Sanyuktha Hansraj**

MSc Data Analytics with Bio AI
Digital University Kerala


GitHub: https://github.com/Sanyuktha-hansraj

---

# 📝 License

This project is intended for educational and research purposes. Please ensure proper attribution and citation of the DDI dataset when using this work.

---

# 🚀 Status

**Active Development**
