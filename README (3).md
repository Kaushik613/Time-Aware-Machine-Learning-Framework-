# EUR/USD Time-Aware ML Framework for Forex Signal Prediction

## Project Overview

This project focuses on predicting daily EUR/USD forex trading signals — **BUY**, **SELL**, or **HOLD** — using a time-aware machine learning approach built on historical price data.

Instead of using raw closing prices alone, we engineer **16 temporal features** that give the model contextual knowledge about past price behaviour, volatility, and momentum.

We implement and compare two approaches:
- **Without Temporal Features** (baseline — yesterday's price only)
- **With Temporal Features** (full 16-feature engineered set)

The goal is to prove that temporal feature engineering significantly improves prediction accuracy for forex signal classification.

---

## Repository Structure

```
eurusd-ml-framework/
│
├── EURUSD_ML_2Models_Final__.ipynb    ← Main Jupyter notebook (run this)
├── EUR_USD_Historical_Data3.csv       ← Historical EUR/USD price dataset
├── README.md                          ← This file
└── graphs/                            ← Auto-created; stores all output graphs
    ├── fig1_price_history.png
    ├── fig2_class_distribution.png
    ├── fig3_daily_returns.png
    ├── fig4_rolling_averages.png
    ├── fig5_momentum.png
    ├── fig6_volatility.png
    ├── fig7_correlation_heatmap.png
    ├── fig8_model_accuracy.png
    ├── fig9_with_vs_without.png
    ├── fig10_confusion_matrices.png
    └── fig11_feature_importance.png
```

---

## Dataset

**EUR/USD Historical Daily Price Data**

| Detail | Info |
|--------|------|
| Source file | `EUR_USD_Historical_Data3.csv` |
| Coverage | January 2001 – February 2025 |
| Records | ~11,284 daily observations |
| Features | Date, Price (close), Open, High, Low, Vol., Change % |
| Target | 3-class signal: BUY / SELL / HOLD |
| Class threshold | ±0.1% next-day price change |

> ⚠️ Place `EUR_USD_Historical_Data3.csv` in the **same folder** as the notebook before running.

---

## Setup Instructions

### 1. Requirements

Make sure you have Python installed (Python 3.8+ recommended).

Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/eurusd-ml-framework.git
cd eurusd-ml-framework
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook EURUSD_ML_2Models_Final__.ipynb
```

> **Using Google Colab instead?**  
> Go to https://colab.research.google.com → File → Upload Notebook → select `EURUSD_ML_2Models_Final__.ipynb`  
> Then upload `EUR_USD_Historical_Data3.csv` using the Files panel on the left sidebar.

---

## How to Run the Code

Open `EURUSD_ML_2Models_Final__.ipynb` and run cells in order from top to bottom.

| Step | Description | Output |
|------|-------------|--------|
| Step 1 | Install libraries | Confirms all packages loaded |
| Step 2 | Load dataset | Shape, column names, date range |
| Step 3 | Temporal feature engineering + BUY/SELL/HOLD labels | 16 new feature columns added |
| Step 4 | Exploratory Data Analysis | 7 graphs saved to `graphs/` |
| Step 5 | Train 2 ML models (Logistic Regression + Random Forest) | Accuracy, F1, classification report |
| Step 6 | WITH vs WITHOUT temporal features comparison | Side-by-side accuracy for both models |
| Step 7 | Result graphs | 4 result graphs saved to `graphs/` |
| Step 8 | Final report | Printed summary of all metrics |

**In Jupyter:** Click `Kernel → Restart & Run All` to run everything at once.  
**In Google Colab:** Click `Runtime → Run all`.

---

## Feature Engineering

Raw OHLC data is enriched with 16 temporal features before training:

| Feature | What it tells the model |
|---------|------------------------|
| `lag_1` – `lag_7` | Closing prices from 1 to 7 days ago |
| `rolling_mean_7` / `rolling_mean_14` | Average price over last 7 or 14 days |
| `rolling_std_7` / `rolling_std_14` | How volatile the market has been |
| `momentum_3` / `momentum_7` | Is price trending up or down? |
| `daily_return` | Percentage price change from yesterday |
| `abs_return` | Absolute size of yesterday's move |
| `range_7` | High minus low over the last 7 days |
| `day_of_week` | Some days are more volatile than others |

### Target Labels

| Label | Value | Condition |
|-------|-------|-----------|
| SELL | 0 | Next-day price falls more than 0.1% |
| HOLD | 1 | Next-day change within ±0.1% |
| BUY | 2 | Next-day price rises more than 0.1% |

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════╗
║         LAYER 1 – DATA INGESTION & PREPARATION              ║
║  EUR_USD_Historical_Data3.csv → Parse Dates → Clean Data    ║
║  → Engineer 16 Temporal Features → Label BUY/SELL/HOLD      ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║
          ┌────────────┴────────────┐
          ▼                         ▼
   [WITHOUT Features]         [WITH Features]
   Baseline: lag_1 only       Full 16-feature set
   (raw price only)           (time-aware context)
          │                         │
          ▼                         ▼
╔══════════════════════════════════════════════════════════════╗
║         LAYER 2 – MODEL TRAINING (80% / 20% Split)         ║
║                                                              ║
║   [Logistic Regression]       [Random Forest]               ║
║   Linear decision boundary    Ensemble of decision trees    ║
║   Fast, interpretable         Non-linear, feature-ranked    ║
╚══════════════════════╦═══════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════╗
║         LAYER 3 – EVALUATION & COMPARISON                   ║
║  Accuracy · Precision · Recall · F1-Score · ROC-AUC        ║
║  WITH vs WITHOUT temporal features — hypothesis proven ✅   ║
╚══════════════════════════════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════╗
║         OUTPUT: BUY / SELL / HOLD Signal                    ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Results Summary

### Model Accuracy: WITH vs WITHOUT Temporal Features

| Model | Without Temporal Features | With Temporal Features | Improvement |
|-------|--------------------------|----------------------|-------------|
| Logistic Regression | Baseline (lag_1 only) | Full feature set | ✅ Higher |
| Random Forest | Baseline (lag_1 only) | Full feature set | ✅ Higher |

> "Temporal feature engineering improves prediction accuracy for both models by providing historical context — price trends, volatility, and momentum — that a single lag variable cannot capture."

### Key Findings

- **Random Forest** outperforms Logistic Regression overall due to its ability to capture non-linear relationships between features
- **Temporal features** consistently improve accuracy across both models — directly proving the project hypothesis
- **Feature importance** analysis (Graph 11) reveals that rolling means and momentum features contribute most to Random Forest predictions
- **HOLD** is the most frequent class due to the ±0.1% threshold, reflecting real market conditions where small daily moves dominate

---

## Key Concepts

| Term | Meaning |
|------|---------|
| **Temporal Feature Engineering** | Creating new input variables from historical time-series data to give the model past context |
| **Lag Variables** | Price values from N days ago — the most direct form of temporal memory |
| **Rolling Statistics** | Averages and standard deviations computed over a sliding time window |
| **Momentum** | The direction and strength of recent price movement |
| **BUY / SELL / HOLD** | 3-class target label derived from next-day price change |
| **Random Forest** | Ensemble of decision trees that votes on the final prediction |
| **Logistic Regression** | Linear model that finds the best boundary between classes |
| **80/20 Time Split** | Train on the first 80% of dates; test on the last 20% — no data leakage |
| **StandardScaler** | Normalises features to zero mean and unit variance before training |
| **F1-Score** | Harmonic mean of Precision and Recall — best metric for imbalanced classes |

---

## Data Note

> This project uses **publicly available** EUR/USD historical market price data. All data consists of aggregated exchange rates — no personal or sensitive information is present.

- No private or user data of any kind is used
- The dataset is freely available from financial data providers
- All processing is done locally — no data is transmitted externally

---

## References

1. Breiman, L. (2001). Random Forests — https://link.springer.com/article/10.1023/A:1010933404324
2. Cox, D.R. (1958). The Regression Analysis of Binary Sequences — https://www.jstor.org/stable/2983890
3. Patel et al. (2015). Predicting Stock and Stochastic Markets Using ML — https://doi.org/10.1016/j.eswa.2014.07.040
4. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python — https://jmlr.org/papers/v12/pedregosa11a.html
5. Murphy, J.J. (1999). Technical Analysis of the Financial Markets. New York Institute of Finance.
6. Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice — https://otexts.com/fpp3/
7. Chawla et al. (2002). SMOTE for Class Imbalance — https://arxiv.org/abs/1106.1813
8. Lundberg & Lee (2017). SHAP Feature Importance — https://arxiv.org/abs/1705.07874

---

*Submitted for 22AIE213 – Machine Learning | Amrita School of Engineering*
