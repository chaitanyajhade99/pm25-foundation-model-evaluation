# Evaluating Zero-Shot Time-Series Foundation Models Versus Domain-Trained Models for Daily $\text{PM}_{2.5}$ Air Quality Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Style](https://img.shields.io/badge/Style-IEEE%20Format-00629B.svg)](https://www.ieee.org/)

This repository provides the official implementation, experimental pipeline, and evaluation framework for benchmarking **zero-shot time-series foundation models (TSFMs)** against domain-trained machine learning architectures and deep neural networks for daily ambient $\text{PM}_{2.5}$ forecasting across Indian metropolitan areas.

---

## 📄 Abstract

Accurate forecasting of ambient fine particulate matter ($\text{PM}_{2.5}$) is critical for public health advisories, urban air quality management, and environmental policy enforcement. While pretrained time-series foundation models (TSFMs) have shown remarkable zero-shot transfer capabilities across synthetic and economic benchmarks, their operational reliability and generalization in highly non-stationary environmental domains—characterized by severe seasonal swings, non-Gaussian sensor noise, missing data, and structural shocks (such as the COVID-19 lockdowns)—remain largely unquantified.

In this work, we conduct a pre-registered comparative evaluation benchmarking leading zero-shot foundation models (**Chronos-Bolt-Small**, **Chronos-Bolt-Base**, and **Chronos-2**) against supervised classical algorithms (**HistGradientBoosting**, **Ridge-AR**), deep sequence models (**multi-horizon pooled LSTM** with learned city entity embeddings), and atmospheric baselines across major Indian metropolitan areas from the Central Pollution Control Board (CPCB) monitoring network. All models are evaluated on a strictly common, leakage-free observation universe across multi-step forecast horizons ($h \in \{1, 3, 7\}$ days) over three annual temporal test folds (2018, 2019, and 2020).

Beyond standard point metrics (MASE, MAE, RMSE) with cluster-bootstrapped 95% confidence intervals, we evaluate:
1. **Extreme Event Detection ($y > P_{90}$):** Precision, Recall, and $F_1$-scores under extreme class imbalance.
2. **Diebold-Mariano Predictive Accuracy Tests:** Pairwise loss differential tests with Harvey-Leybourne-Newbold small-sample correction, Holm-Bonferroni multi-testing control, and Stouffer's pooled meta-analysis across cities.
3. **Distribution-Free Uncertainty Quantification:** Split conformal prediction intervals across diverse environmental regimes (Winter Smog, COVID Lockdown, Non-Winter Baseline).

---

## 🏛️ Methodological Framework & Pipeline Architecture

```mermaid
flowchart TD
    A[CPCB Raw Ambient Air Quality Data\ncity_day.csv] --> B[Pre-Registered Inclusion Audit\nStart <= 2017-07-01 & Completeness >= 89%]
    B --> C[Temporal Partitioning\nRolling Folds: 2018, 2019, 2020\nContext Horizon h in {1, 3, 7}]
    
    C --> D1[Naive & Statistical Baselines\nPersistence, 7-d MA, Climatology, SES]
    C --> D2[Supervised Residual ML\nRidge-AR, HistGradientBoosting]
    C --> D3[Deep Sequence Modeling\nPooled Multi-Horizon LSTM + City Embeddings]
    C --> D4[Zero-Shot Foundation Models\nChronos-Bolt-Small, Chronos-Bolt-Base, Chronos-2]
    
    D1 --> E[Common Row Universe\nStrict Intersection of Valid Forecasts]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F1[Point Accuracy & Skill\nMacro MASE, MAE, RMSE, Skill Score\n95% Cluster-Bootstrap CIs]
    E --> F2[Extreme Event Detection\nThreshold y > P90\nPrecision, Recall, F1]
    E --> F3[Hypothesis Testing\nDiebold-Mariano + Holm Correction\nStouffer's Pooled Meta-Analysis]
    E --> F4[Conformal Uncertainty Quantification\nGlobal, Seasonal, Rolling, Scaled, Native\nTarget Coverage 80% and 90%]
```

---

## 📊 Summary of Evaluated Models

| Model Class | Model Identifier | Optimization / Pretraining Paradigm | Context Window / Input Details |
| :--- | :--- | :--- | :--- |
| **Naive Benchmark** | `persist` | Zero-parameter Random Walk | Ground truth at forecast origin $t$ |
| **Statistical Benchmark** | `ma7` | 7-day Simple Moving Average | Past 7 observed days |
| **Statistical Benchmark** | `clim` | Calendar-month climatology | Long-term training month means |
| **Statistical Benchmark** | `ses` | Simple Exponential Smoothing (ETS(A,N,N)) | Grid search $\alpha \in [0.05, 0.95]$ on train split |
| **Supervised Classical** | `ridge` | L2-regularized linear model | Multi-lag AR features, rolling statistics, seasonality |
| **Supervised Classical** | `hgb` | Histogram-based Gradient Boosting | Nonlinear tree splits with native missing-value routing |
| **Deep Sequence** | `lstm` | Pooled Multi-Horizon LSTM (PyTorch) | Lookback $L=60$, 4D temporal sequence + 4D city embedding |
| **Foundation Model** | `chronos-bolt-small` | Pretrained Transformer (Zero-Shot) | Historical context sequence up to origin $t$ |
| **Foundation Model** | `chronos-bolt-base` | Pretrained Transformer (Zero-Shot) | Historical context sequence up to origin $t$ |
| **Foundation Model** | `chronos-2` | Autoregressive TSFM (Zero-Shot) | Historical context sequence up to origin $t$ |

---

## 🔬 Experimental Protocol & Anti-Leakage Controls

1. **Strict Temporal Isolation:** At forecast origin $t$, all feature extraction, normalizations, and context windows strictly use observations $\tau \le t$. No future information or test-fold statistics leak into model inputs.
2. **Common Row Universe:** An evaluation instance is scored if and only if **every** candidate model produces an admissible forecast for that (city, fold, horizon, date) tuple.
3. **Regime Stratifications:**
   * **Winter Smog:** November 1 to February 28 (severe thermal inversion & crop burning).
   * **Lockdown Shock:** March 25 to May 31, 2020 (national COVID-19 lockdown).
   * **Other:** Standard baseline meteorological conditions.
4. **Spatial Cluster Bootstrap:** Standard i.i.d. resampling artificially deflates standard errors due to spatial autocorrelation across nearby cities. Cluster bootstrapping ($B=1000$) resamples entire cities to yield statistically honest 95% confidence intervals.
5. **Multiple Testing Correction:** All pairwise Diebold-Mariano tests against persistence are adjusted per city via the **Holm-Bonferroni** procedure, and synthesized nationally using **Stouffer's method**.

---

## 🚀 Getting Started & Execution

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with $\ge 12\,\text{GB}$ VRAM recommended (e.g. NVIDIA T4 or A100)

### 2. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/chaitanyajhade99/pm25-foundation-model-evaluation.git
cd pm25-foundation-model-evaluation
pip install -r requirements.txt
```

### 3. Running in Google Colab (Recommended)
1. Open [`pm25_foundation_model_evaluation.ipynb`](pm25_foundation_model_evaluation.ipynb) in Google Colab.
2. Navigate to **Runtime -> Change runtime type** and select **T4 GPU**.
3. Execute all cells sequentially. When prompted in Cell 3, upload `archive.zip` (containing `city_day.csv` from the CPCB Kaggle dataset).
4. The final cell generates all high-resolution figures in `results/` and outputs `summary.txt`.

### 4. Running Locally
Set the path to `city_day.csv` in the configuration dictionary:
```python
CFG["DATA_PATH"] = "path/to/city_day.csv"
```
Execute the notebook via Jupyter Lab or VS Code.

---

## 📑 Repository Structure

```
pm25-foundation-model-evaluation/
├── README.md                              # IEEE-formatted project documentation
├── requirements.txt                       # Python dependency requirements
├── .gitignore                             # Git exclusion configuration
├── pm25_foundation_model_evaluation.ipynb # Main IEEE research paper notebook
└── pm25_tsfm_colab (1).ipynb              # Colab-ready notebook mirror
```

---

## 📖 Citation & References

If you find this research evaluation or codebase useful in your academic work, please consider citing:

```bibtex
@article{pm25_tsfm_evaluation_2026,
  title   = {Evaluating Zero-Shot Time-Series Foundation Models Versus Domain-Trained Models for Daily PM2.5 Air Quality Forecasting},
  author  = {Jhade, Chaitanya and Contributors},
  journal = {IEEE Transactions on Neural Networks and Learning Systems (Preprint)},
  year    = {2026}
}
```

### Key References
- **Chronos:** Ansari et al., *"Chronos: Learning the Language of Time Series,"* Transactions on Machine Learning Research (TMLR), 2024.
- **Conformal Prediction:** Angelopoulos & Bates, *"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,"* arXiv:2107.07511, 2021.
- **Diebold-Mariano Test:** Diebold & Mariano, *"Comparing Predictive Accuracy,"* J. Business & Economic Statistics, 1995.
- **CPCB Data:** Central Pollution Control Board, *"National Ambient Air Quality Status & Trends in India,"* Ministry of Environment, Forest and Climate Change, Govt. of India.
