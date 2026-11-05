# Multi-Feature Stock & Crypto Price Prediction with PyTorch LSTM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A production-grade, modular deep learning framework for **financial time-series forecasting** (Stocks, Crypto, and Forex). Built with PyTorch, it features multi-step trajectory projection, comparative feature ablation benchmarking, live external API/WebSocket ingestion, custom dataset upload support, and AdamW decoupled weight decay regularization.

---

## 🌟 Key Features

- **Multi-Feature Model Architecture**:
  - Compares 4 feature architectures side-by-side:
    1. `Univariate`: Target Close price only.
    2. `Basic Market`: Raw OHLCV (Open, High, Low, Close, Volume).
    3. `Market + Trend`: OHLCV + Moving Averages (`SMA_10`, `SMA_20`, `EMA_12`).
    4. `Full Multi-Feature`: OHLCV + Trend + Momentum (`RSI_14`, `MACD`) + Volatility (`Bollinger Bands`, 20-period Volatility).
- **Multi-Step Ahead Forecasting ($N$ Candles Ahead)**:
  - Direct multi-horizon linear projection head predicting $t+1, t+2, \dots, t+N$ simultaneously.
  - Automatically tracks error degradation curves and forecast cones across time steps.
- **Support for Any Timeframe & Asset Class**:
  - Daily Stocks (`AAPL`, `TSLA`, `NVDA`, `SPY`).
  - 24/7 Intraday Crypto (`BTC-USD`, `ETH-USD`, `SOL-USD`) on `1m`, `5m`, `15m`, `1h` bars.
  - Forex Currencies (`EURUSD=X`, `GBPUSD=X`).
- **Flexible Data Ingestion Space**:
  - Train directly on your own custom datasets (CSV, Excel, Parquet, JSON) with automatic column & timestamp standardizing.
  - Connect your custom broker/exchange REST APIs (Binance, Alpaca, Polygon, Interactive Brokers).
- **Anti-Overfitting Multi-Layer Defense**:
  - **AdamW Optimizer** with true decoupled L2 weight decay regularization (`weight_decay=1e-4`).
  - Layer Normalization, Dropout (0.20), Gradient Norm Clipping (1.0), and Early Stopping with weight restoration.
- **Google Colab Edition**:
  - Standalone, GPU-accelerated notebook (`Stock_LSTM_Colab_Edition.ipynb`) with interactive forms, file upload buttons, and inline plots.

---

## 📁 Repository Structure

```text
├── lstm_model.py                     # Stacked PyTorch LSTM with dynamic multi-step head
├── data_pipeline.py                  # Leak-free window generator & technical indicators
├── train_and_compare.py              # Training loop (AdamW), multi-feature benchmarking & graphs
├── live_inference.py                 # Real-time streaming predictor & trajectory generator
├── main.py                           # CLI pipeline runner
├── train_from_custom_api.py          # Dedicated Phase 1 training from custom API/JSON
├── custom_external_feed_example.py   # Broker API / WebSocket integration template
├── Stock_LSTM_Colab_Edition.ipynb    # Standalone Google Colab GPU notebook
├── sample_custom_dataset.csv         # Reference sample dataset template
└── outputs/                          # Generated comparison charts & model checkpoints (.pt)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_USERNAME/stock_lstm_prediction.git
cd stock_lstm_prediction
pip install torch pandas numpy matplotlib scikit-learn yfinance requests
```

### 2. Train on Public Stocks or Crypto (Single or Multi-Step)

```bash
# Forecast the next 5 hourly candles for Bitcoin
python main.py --symbol BTC-USD --interval 1h --period 60d --forecast_horizon 5

# Forecast the next 10 daily candles for Apple
python main.py --symbol AAPL --interval 1d --period 2y --forecast_horizon 10

# Forecast 5-minute intraday bars for Tesla
python main.py --symbol TSLA --interval 5m --period 5d --forecast_horizon 12
```

### 3. Train on Your Own Custom Dataset (CSV / Excel)

```bash
python main.py --data_path "sample_custom_dataset.csv" --epochs 25 --forecast_horizon 5
```

### 4. Real-Time Live Inference

```python
import pandas as pd
from live_inference import LiveStockPredictor

# Load trained checkpoint
predictor = LiveStockPredictor("./outputs/btc_usd_1h_h5/lstm_basic_market_(ohlcv).pt")

# Ingest live market feed
live_df = pd.read_csv("my_live_stream.csv", index_col="Date", parse_dates=True)
forecast = predictor.predict_from_live_dataframe(live_df)

print(f"Current Price:  ${forecast['current_price']}")
for step in forecast["trajectory"]:
    print(f"Step t+{step['step']}: ${step['predicted_price']} ({step['delta']:+} | {step['pct_change']:+.2f}%) -> {step['signal']}")
```

### 5. Universal Multi-Company Training (Train ONE Model on Entire Folder)

Train **ONE unified, generalized LSTM** across all company CSV files in a folder (e.g. 55 NIFTY 50 companies) with independent per-company scaling and an interactive company explorer:

```bash
# Train one model on all company CSVs in the folder
python multi_company_unified_lstm.py \
    --data_dir "dataset/archive/NifSent/NIFTY 50" \
    --epochs 20 \
    --batch_size 256 \
    --forecast_horizon 5 \
    --select_ticker RELIANCE

# Switch to view any other company stock forecast instantly without retraining
python multi_company_unified_lstm.py --select_ticker TCS --view_only
python multi_company_unified_lstm.py --select_ticker INFY --view_only
```

---

## ☁️ Google Colab Quickstart

Open `Universal_Multi_Company_Stock_LSTM.ipynb` in Google Colab (with T4 GPU runtime enabled):

```bash
# 1. Clone repository
!git clone https://github.com/Raghava-sensai/stock-lstm-prediction.git
%cd stock-lstm-prediction

# 2. Install dependencies
!pip install -q torch pandas numpy matplotlib scikit-learn yfinance ipywidgets

# 3. Train ONE universal model on all 55 companies (GPU)
!python multi_company_unified_lstm.py --data_dir "dataset/archive/NifSent/NIFTY 50" --epochs 20 --batch_size 256

# 4. View interactive dropdown selector
import pickle
from multi_company_unified_lstm import launch_interactive_widget
with open("outputs/universal_multi_company/eval_results.pkl", "rb") as f:
    eval_results = pickle.load(f)
launch_interactive_widget(eval_results)
```

---

## 📊 Benchmark Visualizations

The framework automatically outputs evaluation figures into `./outputs/<asset>_<interval>/`:
- **`feature_comparison_predictions.png`**: Multi-model prediction overlays vs actual prices.
- **`multi_step_trajectories.png`**: Step-by-step error degradation and forecast trajectory cones.
- **`feature_comparison_loss.png`**: Train vs Validation convergence curves across epochs.
- **`feature_comparison_metrics.png`**: Side-by-side RMSE, MAE, and directional hit rate bar charts.
- **`feature_correlations.png`**: Feature correlation matrix with target price.

---

## 📄 License
MIT License. Open-source for personal, educational, and commercial quantitative research.
