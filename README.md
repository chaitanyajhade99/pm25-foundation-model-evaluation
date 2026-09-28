# Beyond MASE: Spike Detection and Interval Calibration in Zero-Shot $\text{PM}_{2.5}$ Forecasting for Indian Cities

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IEEE Style](https://img.shields.io/badge/Format-IEEE%20Research%20Paper-00629B.svg)](https://www.ieee.org/)

**Chaitanya Jhade\*, Atharva Khewle, Prasad Satpute, Hringkesh Singh, Himani Deshpande**  
*Department of Artificial Intelligence & Data Science, Thadomal Shahani Engineering College, Mumbai, India*  
Correspondence: `chaitanyajhade99@gmail.com`, `Himani.deshpande@thadomal.org`

---

## 📄 Abstract

Time-series foundation models such as **Chronos** and **Chronos Bolt** can forecast a new series without being trained on it, which makes them attractive for cities that lack the resources to train and maintain their own air quality models. This paper evaluates three zero-shot Chronos family models against seven trained and naive baselines for daily $\text{PM}_{2.5}$ forecasting across nine Indian cities, using rolling-origin, time-ordered splits over three test years including the 2020 lockdown period. 

Beyond the mean absolute scaled error (MASE) that most forecasting papers report, we measure two properties that matter for a public health deployment: **how often each model detects a day that exceeds the city's own high pollution threshold**, and **how often its uncertainty interval actually contains the true value**, broken down by season, lockdown status, and pollution level. 

The best zero-shot model, **Chronos 2**, achieves the lowest average error at every horizon and about nineteen percent lower error than persistence at seven days ahead, yet at one day ahead its ability to flag high pollution days is well behind simple persistence, and every model's default uncertainty interval covers only about half of true high pollution days rather than the intended eighty percent. A **level-scaled conformal calibration step** restores interval coverage on these days to around seventy percent or higher across models and horizons, with one case (gradient boosting at seven days) falling just under at sixty-nine percent. The results indicate that **average error and public health usefulness are not the same thing**, and that reporting either alone is not enough for city-scale deployment decisions.

**Index Terms:** Air quality forecasting, conformal prediction, foundation models, $\text{PM}_{2.5}$, time series forecasting, zero-shot learning.

---

## 🎯 The Core Thesis: Beyond Average Error (MASE)

Standard forecasting benchmarks evaluate models using aggregate point-accuracy metrics such as Mean Absolute Scaled Error (MASE) or Root Mean Squared Error (RMSE). In an operational public health context, however, these measures treat every day equally:

1. **The Spike Detection Dilemma:** Alerts and emergency interventions (e.g., school closures, vehicle restrictions, anti-smog guns) are triggered specifically on days when pollution spikes. A model can achieve superior aggregate MASE while missing the vast majority of extreme pollution events.
2. **The Uncertainty Breakdown on Critical Days:** Standard or global conformal prediction intervals suffer severe miscoverage during winter smog episodes and spike days, covering only 36–54% of observations instead of the nominal 80%.
3. **The Multi-Horizon Tradeoff:** Persistence is hard to beat at 1-day ahead because $\text{PM}_{2.5}$ is heavily autocorrelated. The true advantage of foundation models only emerges at longer horizons (3 and 7 days).

---

## 🏛️ Methodological Framework & Pipeline Architecture

```mermaid
flowchart TD
    A[CPCB Ground Monitoring Data\ncity_day.csv: 29,531 daily records across 26 cities] --> B[Pre-Registered City Audit\nStart <= 2017-07-01 & Coverage >= 89%\nRetained: 9 Cities | Excluded: Mumbai, etc.]
    B --> C[Rolling Time-Ordered Splits\nTrain: History up to fold\nCal: 12 months preceding test\nTest: 2018, 2019, 2020 Lockdown]
    
    C --> D1[Naive Baselines\nPersistence, 7-d MA, Climatology]
    C --> D2[Trained Baselines\nSES, Ridge-AR, HistGradientBoosting, Pooled LSTM]
    C --> D3[Zero-Shot Foundation Models\nChronos-Bolt-Small, Chronos-Bolt-Base, Chronos-2]
    
    D1 --> E[Common Row Universe\nIntersection of valid forecasts across all 9 models]
    D2 --> E
    D3 --> E
    
    E --> F1[1. Point Accuracy & Skill\nMacro MASE, MAE, RMSE\nDiebold-Mariano + Stouffer Pooled Test]
    E --> F2[2. Event Spike Detection\nCity-specific P90 Threshold\nRecall, Precision, F1 Score]
    E --> F3[3. Conformal Interval Calibration\nGlobal, Seasonal, Rolling, Level-Scaled, Native\nStratified: All, Winter, Lockdown, High-Pollution]
```

---

## 🏙️ Dataset & Pre-Registered City Selection

Using the Central Pollution Control Board (CPCB) continuous ambient monitoring records (`city_day.csv`, 2015–2020), cities were selected **prior to running any experiments** based on two objective criteria:
1. Continuous monitoring began on or before **1 July 2017** (giving $\ge 3$ full years before the first test fold).
2. Overall non-missing $\text{PM}_{2.5}$ coverage $\ge \mathbf{89\%}$.

Nine of the twenty-six candidate cities met both criteria. Mumbai was excluded due to only 39% recorded coverage.

### Table 1: Retained Cities and Selection Criteria

| City | History From | $\text{PM}_{2.5}$ Coverage | Status |
| :--- | :---: | :---: | :---: |
| **Delhi** | Jan 2015 | **99.9%** | Retained |
| **Jaipur** | Jun 2017 | **98.9%** | Retained |
| **Thiruvananthapuram** | Jun 2017 | **96.2%** | Retained |
| **Lucknow** | Jan 2015 | **94.9%** | Retained |
| **Hyderabad** | Jan 2015 | **94.3%** | Retained |
| **Chennai** | Jan 2015 | **94.2%** | Retained |
| **Bengaluru** | Jan 2015 | **92.7%** | Retained |
| **Gurugram** | Nov 2015 | **90.8%** | Retained |
| **Amritsar** | Feb 2017 | **89.5%** | Retained |

---

## 🔬 Benchmark Models

| Model Class | Identifier | Formulation & Architecture | Context / Input Details |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | `Persistence` | Repeats last observed $\text{PM}_{2.5}$, forward-filled across gaps | Last observed day |
| **Naive Baseline** | `Seven day average` | Simple moving average over past 7 observed days | Past 7 days |
| **Naive Baseline** | `Monthly climatology` | Calendar-month historical average from training period | Calendar month index |
| **Statistical Baseline** | `Exponential smoothing` | SES / $\text{ETS}(A,N,N)$, $\alpha \in [0.05, 0.95]$ tuned on train split | Full training history |
| **Supervised Classical** | `Ridge regression` | Linear shrinkage on lags, rolling stats, seasonality; predicts residual to persistence | Lags $\{1,2,3,6,13,20,27\}$, rolling stats |
| **Supervised Classical** | `Gradient boosting` | HistGradientBoosting (300 trees, lr=0.05, min_samples=30); predicts residual to persistence | Lags, rolling stats, seasonal encodings |
| **Deep Learning** | `LSTM` | Single-layer LSTM ($H=64$), pooled over 9 cities with learned 4D city embedding; residual head | Lookback $L=60$, 4D inputs $(\tilde{y}, m, a, \text{miss}_{30})$ |
| **Foundation Model** | `Chronos Bolt Small` | Pretrained patch-based encoder-decoder transformer (zero-shot) | Up to 512 context days |
| **Foundation Model** | `Chronos Bolt Base` | Pretrained patch-based encoder-decoder transformer (zero-shot) | Up to 512 context days |
| **Foundation Model** | `Chronos 2` | Pretrained causal foundation model with continuous tokenization (zero-shot) | Up to 512 context days |

---

## 📊 Experimental Results

### Table 2: Mean Absolute Scaled Error (MASE) Across 9 Cities
*Evaluated across 20 valid city-fold combinations. Values $< 1.0$ indicate performance superior to in-sample one-step persistence.*

| Model | $h = 1$ day | $h = 3$ days | $h = 7$ days |
| :--- | :---: | :---: | :---: |
| **Chronos 2** | **0.600** | **0.562** | **0.537** |
| **Chronos Bolt Base** | 0.611 | 0.575 | 0.548 |
| **Chronos Bolt Small** | 0.616 | 0.576 | 0.550 |
| **Gradient boosting** | 0.604 | 0.573 | 0.561 |
| **Ridge regression** | 0.611 | 0.581 | 0.563 |
| **LSTM** | 0.607 | 0.588 | 0.577 |
| **Exponential smoothing** | 0.648 | 0.607 | 0.582 |
| **Persistence** | 0.653 | 0.676 | 0.676 |
| **Seven day average** | 0.772 | 0.627 | 0.602 |
| **Monthly climatology** | 1.489 | 1.061 | 0.939 |

> **Key Finding 1:** Chronos 2 achieves the lowest MASE at all horizons. Its margin over persistence widens from **7.3%** at $h=1$ to **16.2%** at $h=3$ and **19.4%** at $h=7$. While only 2 of 9 cities show individual significance at $h=1$ under Holm-adjusted Diebold-Mariano tests, Stouffer's pooled test confirms a highly significant consistent advantage across cities ($z=7.79, p < 0.001$).

---

### Table 3: High Pollution Day Detection ($y > P_{90}$) at $h = 1$ Day
*Threshold defined as each city's training-period 90th percentile ($N=206$ positive days out of 5,626 pooled test days, 3.7%).*

| Model | Recall | Precision | $F_1$ Score |
| :--- | :---: | :---: | :---: |
| **Persistence** | **0.428** | 0.419 | **0.494** |
| **Exponential smoothing** | 0.389 | 0.395 | 0.456 |
| **Chronos Bolt Small** | 0.302 | 0.402 | 0.392 |
| **Chronos 2** | 0.240 | **0.437** | 0.401 |
| **Chronos Bolt Base** | 0.262 | 0.424 | 0.373 |
| **Gradient boosting** | 0.214 | 0.426 | 0.365 |
| **LSTM** | 0.187 | 0.412 | 0.336 |

> **Key Finding 2:** The model with the lowest average error (Chronos 2) is **not** the best at flagging high-pollution days. At 1 day ahead, persistence achieves the highest recall (42.8%) and highest $F_1$ (0.494). Chronos 2 achieves the highest precision (43.7%) but catches only 24.0% of spikes. At $h=7$, gradient boosting leads in $F_1$ (0.201), while Chronos 2 recall drops to 0.2%.

---

### Table 4: Nominal 80% Interval Coverage by Environmental Regime
*Comparison of Global Conformal, Level-Scaled Conformal, and TSFM Native Quantiles across All Days, Winter Smog, COVID Lockdown, and High-Pollution Days.*

| Model & Calibration Scheme | All Days | Winter Smog | COVID Lockdown | High Pollution ($y > P_{90}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient boosting, global ($h=1$)** | 0.84 | 0.71 | 0.94 | **0.43** |
| **Gradient boosting, scaled ($h=1$)** | 0.82 | 0.83 | 0.79 | **0.76** |
| **Chronos 2, global ($h=1$)** | 0.85 | 0.72 | 0.94 | **0.46** |
| **Chronos 2, native ($h=1$)** | 0.80 | 0.81 | 0.78 | **0.65** |
| **Chronos 2, scaled ($h=1$)** | 0.81 | 0.81 | 0.79 | **0.72** |
| **Gradient boosting, global ($h=7$)** | 0.85 | 0.71 | 0.94 | **0.45** |
| **Gradient boosting, scaled ($h=7$)** | 0.81 | 0.82 | 0.81 | **0.69** |
| **Chronos 2, global ($h=7$)** | 0.85 | 0.73 | 0.94 | **0.51** |
| **Chronos 2, native ($h=7$)** | 0.78 | 0.80 | 0.75 | **0.76** |
| **Chronos 2, scaled ($h=7$)** | 0.81 | 0.82 | 0.80 | **0.72** |

> **Key Finding 3:** Standard global conformal intervals appear well-calibrated overall (~84–85%), but suffer a catastrophic failure on high-pollution days, covering only **36–54%** of true observations. 
> 
> The proposed **level-scaled conformal calibration** (which scales residuals by $\max(\hat{y}, 10\,\mu\text{g}/\text{m}^3)$) restores high-pollution coverage to **72–76%**, while maintaining nominal ~81–83% coverage during winter smog without requiring explicit seasonal stratification.

---

### 📉 Impact of Sensor Missingness

When data in the 30 days preceding the forecast origin is $>25\%$ missing, MASE increases by $2\times$ to $3\times$ across all models. Persistence suffers the greatest absolute degradation, whereas Chronos foundation models degrade proportionally the least, benefiting from their long effective context window (up to 512 days).

---

## 🚀 Execution & Reproducibility

### 1. Requirements & Setup
```bash
git clone https://github.com/chaitanyajhade99/pm25-foundation-model-evaluation.git
cd pm25-foundation-model-evaluation
pip install -r requirements.txt
```

### 2. Google Colab (Recommended)
1. Open [`pm25_foundation_model_evaluation.ipynb`](pm25_foundation_model_evaluation.ipynb) in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Run all cells. Upload `archive.zip` (containing `city_day.csv` from Kaggle's *Air Quality Data in India, 2015 to 2020*) when prompted.
4. Summary metrics, tables, and figures are automatically saved to `results/`.

---

## 📑 Repository Contents

```
pm25-foundation-model-evaluation/
├── README.md                              # Complete paper documentation & benchmark results
├── requirements.txt                       # Core Python dependencies
├── .gitignore                             # Ignored checkpoints and artifacts
├── pm25_foundation_model_evaluation.ipynb # Main research paper notebook (IEEE format)
└── pm25_tsfm_colab (1).ipynb              # Mirror notebook for Colab workflows
```

---

## 📖 Citation

If you use this evaluation framework, conformal calibration protocol, or benchmark results in your research, please cite:

```bibtex
@article{jhade2026beyondmase,
  title   = {Beyond MASE: Spike Detection and Interval Calibration in Zero-Shot PM2.5 Forecasting for Indian Cities},
  author  = {Jhade, Chaitanya and Khewle, Atharva and Satpute, Prasad and Singh, Hringkesh and Deshpande, Himani},
  journal = {arXiv preprint / IEEE Conference Submission},
  year    = {2026}
}
```

### References Cited
1. A. F. Ansari et al., "Chronos: Learning the language of time series," *arXiv:2403.07815*, 2024.
2. A. Das, W. Kong, R. Sen, and Y. Zhou, "A decoder only foundation model for time series forecasting," *arXiv:2310.10688*, 2023.
3. Amazon Web Services, "Fast and accurate zero shot forecasting with Chronos Bolt and AutoGluon," *AWS ML Blog*, Nov. 2024.
4. A. F. Ansari et al., "Chronos 2: From univariate to universal forecasting," *arXiv:2510.15821*, 2025.
5. Y. Huang, L. Jiang, and Z. Y. Liu, "Evaluating the generalizability of foundation models for extreme environmental events: case study of California wildfire PM2.5," *arXiv:2607.07951*, 2026.
6. R. Bharadwaj, M. Gupta, and P. Arjunan, "Air Quality Arena: a large scale multi region ground monitoring dataset and benchmark for air quality forecasting with time series foundation models," *arXiv:2607.19381*, 2026.
7. M. Krishan et al., "Air quality modelling using long short term memory (LSTM) over NCT Delhi, India," *Air Qual. Atmos. Health*, vol. 12, no. 8, pp. 899–908, 2019.
8. J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman, "Distribution free predictive inference for regression," *J. Amer. Statist. Assoc.*, vol. 113, no. 523, pp. 1094–1111, 2018.
9. V. Vovk, A. Gammerman, and G. Shafer, *Algorithmic Learning in a Random World*. Springer, 2005.
10. I. Gibbs and E. Candes, "Adaptive conformal inference under distribution shift," in *Proc. NeurIPS*, vol. 34, 2021.
11. R. Rao, "Air Quality Data in India (2015 to 2020)," *Kaggle*, 2020.
12. F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *J. Bus. Econ. Statist.*, vol. 13, no. 3, pp. 253–263, 1995.
13. D. Harvey, S. Leybourne, and P. Newbold, "Testing the equality of prediction mean squared errors," *Int. J. Forecast.*, vol. 13, no. 2, pp. 281–291, 1997.
14. J. H. Friedman, "Greedy function approximation, a gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.
15. S. Hochreiter and J. Schmidhuber, "Long short term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.
16. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
17. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, vol. 32, 2019.
