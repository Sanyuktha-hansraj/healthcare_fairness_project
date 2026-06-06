# 🏥 Healthcare Fairness ML Project
## Detecting Algorithmic Bias in Skin Lesion Classification

A comprehensive framework for evaluating and mitigating fairness issues in healthcare AI, specifically focused on detecting bias in skin lesion malignancy prediction across different skin tone groups using the Diverse Dermatology Images (DDI) dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Fairness Metrics](#fairness-metrics)
- [Results](#results)
- [Dashboard](#dashboard)
- [Contributing](#contributing)

---

## 🎯 Overview

This project investigates **algorithmic bias in dermatological AI systems**, specifically examining whether skin lesion classification models perform equally well across different skin tone groups. Using the Fitzpatrick scale classification, we analyze fairness across three demographic groups:

- **Fitzpatrick I-II** (Light skin)
- **Fitzpatrick III-IV** (Medium skin)
- **Fitzpatrick V-VI** (Dark skin)

The analysis includes multiple fairness metrics, bias detection, and mitigation strategies to ensure equitable healthcare outcomes.

---

## ⚠️ Problem Statement

Healthcare AI systems have been shown to exhibit disparate performance across demographic groups, leading to potential healthcare inequities. Dermatological AI is particularly susceptible to bias due to:

1. **Dataset bias** - underrepresentation of darker skin tones in training data
2. **Model performance disparity** - lower accuracy for underrepresented groups
3. **Clinical impact** - missed diagnoses can lead to serious health consequences

This project quantifies and addresses these fairness issues in skin lesion classification.

---

## ✨ Key Features

- **Fairness-Focused Analysis**: Multiple fairness metrics including:
  - Demographic Parity
  - Equal Opportunity
  - Equalized Odds
  - Calibration Within Groups

- **Comprehensive EDA**: Exploratory data analysis with demographic breakdowns and outcome disparities

- **Configurable Pipeline**: YAML-based configuration for easy customization of metrics and thresholds

- **Interactive Dashboard**: Streamlit-based dashboard for visualizing fairness metrics and model performance

- **Feature Engineering**: Automated extraction of image statistics and metadata features

- **Extensible Framework**: Modular design for adding new metrics and mitigation strategies

---

## 📊 Dataset

**Diverse Dermatology Images (DDI)**

- **Total Images**: 656 dermoscopy images
- **Classes**: Malignant (26.1%) and Benign (73.9%) skin lesions
- **Diversity**: Balanced representation across Fitzpatrick skin tone groups
- **Format**: PNG images with associated metadata

### Skin Tone Distribution
```
Fitzpatrick I-II    (208 images)
Fitzpatrick III-IV  (241 images)
Fitzpatrick V-VI    (207 images)
```

**Note**: This project uses the DDI dataset. Ensure you have proper permissions and cite the original dataset in your work.

---

## 📁 Project Structure

```
fairness_project/
├── config/                          # Configuration files
│   ├── fairness.yml                # Fairness metrics & thresholds
│   └── features.yml                # Feature engineering config
├── src/                            # Source code
│   ├── data_loader.py             # Dataset loading utilities
│   ├── fairness_metrics.py        # Fairness evaluation classes
│   ├── features.py                # Feature extraction pipeline
│   └── __init__.py
├── dashboard/                       # Streamlit dashboard
│   ├── app.py                      # Main dashboard app
│   └── pages/                      # Multi-page dashboard
│       └── 01_eda_cohort.py
├── notebooks/                       # Jupyter notebooks
│   └── 02_eda.ipynb               # Exploratory data analysis
├── data/                           # Data directory
│   └── processed/                  # Processed data & features
│       └── feature_list.json
├── DDI/                            # Dataset location
│   ├── ddi_metadata.csv           # Dataset metadata
│   └── images/                     # Dermoscopy images
├── reports/                        # Analysis reports & logs
│   ├── log.md                     # Week 1 analysis log
│   ├── week2_log.md               # Week 2 progress
│   └── figures/                    # Saved visualizations
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip or conda
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fairness_project
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the project**
   - Ensure your DDI dataset is in the `DDI/` directory
   - Review and customize `config/fairness.yml` if needed

---

## 💻 Usage

### 1. Exploratory Data Analysis
Run the EDA notebook to understand dataset characteristics:
```bash
jupyter notebook notebooks/02_eda.ipynb
```

### 2. Load Dataset
```python
from src.data_loader import DDI_DataLoader

loader = DDI_DataLoader(root_dir="./DDI")
df = loader.get_data()
```

### 3. Calculate Fairness Metrics
```python
from src.fairness_metrics import FairnessEvaluator

evaluator = FairnessEvaluator(
    y_true=y_true,
    y_pred=y_pred,
    y_prob=y_proba,
    sensitive_attr=skin_tone_groups
)

dpd = evaluator.demographic_parity_difference()
eod = evaluator.equal_opportunity_difference()
```

### 4. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- Model performance by skin tone group
- Fairness metric visualizations
- Outcome disparity analysis
- Detailed cohort comparisons

---

## 📐 Fairness Metrics

### Demographic Parity
**Definition**: Equal positive prediction rates across groups
- **Formula**: $\max(P(\hat{Y}=1|A=a)) - \min(P(\hat{Y}=1|A=a))$
- **Threshold**: < 0.1
- **Interpretation**: Measures if model predicts positive outcomes equally for all groups

### Equal Opportunity
**Definition**: Equal true positive rates (sensitivity) across groups
- **Formula**: $\max(TPR_a) - \min(TPR_a)$
- **Threshold**: < 0.1
- **Clinical Importance**: Ensures all skin tones have equal detection of malignant lesions

### Equalized Odds
**Definition**: Equal TPR and FPR across groups
- **Formula**: $\max(TPR_{diff}, FPR_{diff})$
- **Threshold**: < 0.1
- **Interpretation**: Balances both error types across demographic groups

### Calibration Within Groups
**Definition**: Predicted probabilities match actual outcomes for each group
- **Formula**: $E[Y|\hat{Y}=p,A=a] \approx p$ for all groups
- **Threshold**: < 0.05
- **Importance**: Model confidence should be reliable for all skin tones

---

## 📈 Results Summary

### Week 2 Analysis Findings

**Dataset Characteristics:**
- Total images analyzed: 656
- Class distribution: 26.1% malignant, 73.9% benign
- All demographic groups have adequate sample sizes (>30)

**Fairness Observations:**
- Outcome disparity between groups: **7.52 percentage points**
- Fitzpatrick III-IV group has higher malignancy rate (30.71%) vs. others (~23%)
- Ready for fairness-aware model development and mitigation strategies

### Group Performance Breakdown
| Skin Tone Group | Total Images | Malignant | Malignancy Rate |
|---|---|---|---|
| Fitzpatrick I-II | 208 | 49 | 23.56% |
| Fitzpatrick III-IV | 241 | 74 | 30.71% |
| Fitzpatrick V-VI | 207 | 48 | 23.19% |

---

## 📊 Dashboard

The Streamlit dashboard provides interactive visualization of:

1. **Main Dashboard** (`app.py`)
   - Project overview
   - Quick statistics
   - Navigation to analysis pages

2. **Cohort Analysis** (`pages/01_eda_cohort.py`)
   - Demographic breakdowns
   - Fairness metric comparisons
   - Model performance by group
   - Statistical testing

Access the dashboard:
```bash
streamlit run dashboard/app.py
```

---

## 🔧 Configuration

### Fairness Configuration (`config/fairness.yml`)

Customize fairness thresholds and metric parameters:

```yaml
fairness_metrics:
  demographic_parity:
    threshold: 0.1
    
  equal_opportunity:
    threshold: 0.1
    
  equalized_odds:
    threshold: 0.1
    
  calibration_within_groups:
    threshold: 0.05
```

### Feature Configuration (`config/features.yml`)

Define which features to extract and use for modeling.

---

## 📚 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥2.0.0 | Data processing |
| numpy | ≥1.24.0 | Numerical computing |
| PyTorch | ≥2.0.0 | Deep learning |
| scikit-learn | ≥1.3.0 | ML & metrics |
| fairlearn | ≥0.9.0 | Fairness algorithms |
| Streamlit | ≥1.25.0 | Dashboard |
| matplotlib/seaborn | Latest | Visualization |

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Additional fairness metrics (e.g., Disparate Impact, Individual Fairness)
- [ ] Bias mitigation techniques (reweighting, threshold optimization, debiasing algorithms)
- [ ] Model implementations with fairness constraints
- [ ] Extended demographic attributes beyond skin tone
- [ ] Statistical testing suite
- [ ] Performance optimization

---

## 📖 References & Citations

### Key Papers
- Buolamwini, B., & Gebru, T. (2018). "Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification". Conference on Fairness, Accountability and Transparency.
- Tan, B. L., et al. (2022). "Fitzpatrick Scale and Dermatology Research: Concerns and Recommendations".

### Datasets
- Diverse Dermatology Images (DDI) Dataset

### Tools & Libraries
- [fairlearn](https://fairlearn.org/) - Microsoft's fairness toolkit
- [Streamlit](https://streamlit.io/) - Dashboard framework

---

## 📝 License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## 👤 Contact & Support

For questions, issues, or suggestions, please [create an issue](../../issues) or contact the project maintainers.

---

## 📌 Citation

If you use this project in your work, please cite:

```
@project{healthcarefairness2025,
  title={Healthcare Fairness ML Project},
  subtitle={Detecting Algorithmic Bias in Skin Lesion Classification},
  year={2025}
}
```

---

**Last Updated**: 2025-11-24  
**Project Status**: Active Development
