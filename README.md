# Advanced Time Series Forecasting: LSTM vs Attention-LSTM

## Project Overview
This project implements **multivariate time series forecasting** using two deep learning models:

1. **Baseline LSTM** – Standard LSTM model
2. **Attention LSTM** – LSTM with additive (Bahdanau-style) attention mechanism over time steps  

The goal is to capture **long-term dependencies** in non-stationary, seasonal time series and compare performance against a baseline.

---

## Dataset

- **UCI Air Quality Dataset (AirQualityUCI)**  
- Contains multiple sensor measurements over time.  
- Missing values (`-200`) are cleaned using interpolation and forward/backfill.  
- CSV is saved locally for reproducibility.

**Automatic download:** The script downloads the dataset from:  
[AirQualityUCI Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)

---

## Project Structure

TimeSeriesProject/
├─ main.py                  # Full pipeline: data loading, preprocessing, model training, evaluation
├─ src/                     # Optional: if you split code into modules
│   ├─ data_prep.py         # Data cleaning, feature prep, sequences
│   ├─ models.py            # LSTM and Attention LSTM definitions
│   ├─ train.py             # Training loops, evaluation functions
│   └─ utils.py             # Helpers: CSV saving, EDA, plotting
├─ data/
│   └─ air_quality.csv      # Local dataset (auto-download if missing)
├─ outputs/                 # All generated outputs
│   ├─ baseline_predictions.csv
│   ├─ attention_predictions.csv
│   ├─ metrics.csv
│   ├─ attention_weights.csv
│   ├─ attention_summary.txt
│   └─ (plots, EDA figures)
├─ reports/
│   ├─ EDA_summary.txt      # Descriptive stats + plots
│   ├─ model_comparison.md  # Textual analysis & RMSE/MAE/MAPE discussion
│   └─ attention_analysis.md # Interpretation of attention weights
├─ requirements.txt         # All Python dependencies
└─ README.md                # Project explanation, usage, results



---

## How to Run

1. **Mount Google Drive** (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')

2. Run the main script

!python main.py

Optional arguments:

--data_path : path to local CSV
--seq_len   : input lookback window (default 24)
--horizon   : forecast horizon (default 1)
--epochs    : training epochs (default 20)
--batch_size: batch size (default 64)
--hidden    : LSTM hidden units (default 64)
--layers    : number of LSTM layers (default 2)
--lr        : learning rate (default 0.001)
--dropout   : dropout (default 0.1)
--patience  : early stopping patience (default 5)


Outputs

All outputs are saved in outputs/ folder:

baseline_predictions.csv – LSTM predictions

attention_predictions.csv – Attention LSTM predictions

metrics.csv – RMSE, MAE, MAPE for both models

attention_weights.csv – Attention weights for test set

attention_summary.txt – Top-attended timesteps per sample

Results
Model	RMSE	MAE	MAPE (%)
Baseline LSTM	...	...	...
Attention LSTM	...	...	...

Plots and attention heatmaps can be generated directly from the notebook.

Notes

The project demonstrates sequence modeling, attention interpretability, and comparative evaluation.

You can extend this by adding Transformer models for comparison.