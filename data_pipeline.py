"""
Data fetching, Feature Engineering, Preprocessing, and Sliding Window Generator
for Stock Price Prediction.
"""

import os
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_custom_dataset(file_path: str) -> pd.DataFrame:
    """
    Load and standardize a user-uploaded custom dataset from CSV, TSV, Excel, Parquet, or JSON.
    Auto-detects Datetime index, standardizes OHLCV column names, and sorts chronologically.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom dataset file not found at: '{file_path}'")

    ext = os.path.splitext(file_path)[-1].lower()
    print(f"[*] Ingesting custom dataset from: {file_path} ({ext})")

    if ext in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
    elif ext == ".tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
    elif ext == ".json":
        df = pd.read_json(file_path)
    else:
        df = pd.read_csv(file_path)

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]

    # Detect Datetime column
    date_candidates = ["date", "datetime", "timestamp", "time", "date_time", "dt"]
    matched_date_col = None
    for col in df.columns:
        if col.lower() in date_candidates:
            matched_date_col = col
            break

    if matched_date_col:
        df["Datetime"] = pd.to_datetime(df[matched_date_col], errors="coerce")
        df = df.set_index("Datetime")
        if matched_date_col != "Datetime" and matched_date_col in df.columns:
            df = df.drop(columns=[matched_date_col])
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            print("[!] Notice: No timestamp column detected. Generating sequential datetime index.")
            df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="D")

    # Standardize OHLCV columns (case-insensitive)
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ["open", "o"]: col_map[col] = "Open"
        elif cl in ["high", "h"]: col_map[col] = "High"
        elif cl in ["low", "l"]: col_map[col] = "Low"
        elif cl in ["close", "c", "price", "last", "settle", "adj close", "adj_close"]:
            if "Close" not in col_map.values():
                col_map[col] = "Close"
        elif cl in ["volume", "vol", "v", "qty", "quantity"]: col_map[col] = "Volume"

    df = df.rename(columns=col_map)

    # Ensure target Close is present
    if "Close" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"[!] Target 'Close' column missing. Using numeric column '{numeric_cols[0]}' as 'Close'.")
            df["Close"] = df[numeric_cols[0]]
        else:
            raise ValueError(f"Could not find a valid price column. Found columns: {list(df.columns)}")

    # Synthesize secondary OHLCV columns if absent
    if "Open" not in df.columns: df["Open"] = df["Close"]
    if "High" not in df.columns: df["High"] = df["Close"]
    if "Low" not in df.columns: df["Low"] = df["Close"]
    if "Volume" not in df.columns: df["Volume"] = 1.0

    # Ensure numeric types
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index().dropna(subset=["Close"])
    print(f"[+] Custom dataset loaded successfully: {len(df)} rows from {df.index[0]} to {df.index[-1]}.")
    return df


class StockDataset(Dataset):
    """PyTorch Dataset wrapper for sequence time-series data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def fetch_stock_data(
    symbol: str = "AAPL",
    period: str = "2y",
    interval: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance or external sources.
    Args:
        symbol: Ticker symbol, e.g. 'AAPL', 'TSLA', 'NVDA', 'SPY'
        period: 1mo, 3mo, 6mo, 1y, 2y, 5y, max (if start_date not provided)
        interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
    """
    # Automatically clamp period for intraday intervals to avoid Yahoo Finance API errors
    interval_caps = {
        "1m": "7d",
        "2m": "60d",
        "5m": "60d",
        "15m": "60d",
        "30m": "60d",
        "90m": "60d",
        "1h": "730d",
        "60m": "730d",
    }
    if interval in interval_caps and not (start_date and end_date):
        long_periods = ["max", "10y", "5y", "2y", "1y"]
        if interval in ["1m"] and period in long_periods:
            print(f"[*] Notice: Clamping period to '{interval_caps[interval]}' for {interval} interval.")
            period = interval_caps[interval]
        elif interval in ["2m", "5m", "15m", "30m", "90m"] and period in ["max", "10y", "5y", "2y", "1y"]:
            print(f"[*] Notice: Clamping period to '{interval_caps[interval]}' for {interval} interval.")
            period = interval_caps[interval]
        elif interval in ["1h", "60m"] and period in ["max", "10y", "5y"]:
            print(f"[*] Notice: Clamping period to '{interval_caps[interval]}' for {interval} interval.")
            period = interval_caps[interval]

    print(f"[*] Fetching live/historical data for {symbol} (period={period}, interval={interval})...")
    ticker = yf.Ticker(symbol)
    if start_date and end_date:
        df = ticker.history(start=start_date, end=end_date, interval=interval)
    else:
        df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. Verify the symbol and parameters.")

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean index
    df = df.dropna()
    print(f"[+] Loaded {len(df)} records from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}.")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute essential technical indicators for feature comparison:
    - Moving averages (SMA, EMA)
    - Momentum (RSI)
    - MACD & Signal line
    - Bollinger Bands
    - Volatility & Daily returns
    """
    data = df.copy()

    # 1. Moving Averages
    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    # 2. MACD (Moving Average Convergence Divergence)
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # 3. RSI (Relative Strength Index - 14 periods)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # 4. Bollinger Bands (20 periods, 2 std dev)
    rolling_mean_20 = data["Close"].rolling(window=20).mean()
    rolling_std_20 = data["Close"].rolling(window=20).std()
    data["BB_Upper"] = rolling_mean_20 + (2 * rolling_std_20)
    data["BB_Lower"] = rolling_mean_20 - (2 * rolling_std_20)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (rolling_mean_20 + 1e-9)

    # 5. Volatility & Returns
    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()

    # Drop initial NaN rows created by rolling windows
    data = data.dropna()
    return data


def create_sliding_sequences(
    data_matrix: np.ndarray,
    target_idx: int,
    seq_length: int = 30,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sliding window sequences for LSTM:
    X: shape (num_samples, seq_length, num_features)
    y: shape (num_samples, forecast_horizon) -> predicts next N time steps' target values
    """
    X, y = [], []
    num_records = len(data_matrix)
    for i in range(num_records - seq_length - forecast_horizon + 1):
        X.append(data_matrix[i : i + seq_length, :])
        y.append(data_matrix[i + seq_length : i + seq_length + forecast_horizon, target_idx])
    return np.array(X), np.array(y)


def prepare_datasets(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Close",
    seq_length: int = 30,
    forecast_horizon: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict:
    """
    Preprocess, split by time, scale without data leakage, and return sequence arrays + scalers.
    Supports multi-step forecasting for N candles ahead.
    """
    if target_col not in feature_cols:
        raise ValueError(f"Target column '{target_col}' must be included in feature_cols.")

    target_idx = feature_cols.index(target_col)
    raw_data = df[feature_cols].values

    # Strict chronological split: fit scalers ONLY on training rows to guarantee ZERO data leakage
    n = len(raw_data)
    train_end = int(n * train_ratio)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(raw_data[:train_end])
    scaled_data = feature_scaler.transform(raw_data)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.min_ = np.array([feature_scaler.min_[target_idx]])
    target_scaler.scale_ = np.array([feature_scaler.scale_[target_idx]])

    # Build sequence windows across continuous timeline
    all_X, all_y = create_sliding_sequences(scaled_data, target_idx, seq_length, forecast_horizon)

    num_samples = len(all_X)
    if num_samples < 5:
        raise ValueError(
            f"Not enough data samples ({num_samples}) to build sequences. "
            f"Total rows: {n}, seq_length: {seq_length}, forecast_horizon: {forecast_horizon}."
        )

    # Split sequences chronologically
    train_count = int(num_samples * train_ratio)
    val_count = max(1, int(num_samples * val_ratio))
    if train_count + val_count >= num_samples:
        train_count = max(1, num_samples - val_count - 1)

    test_start = train_count + val_count

    X_train, y_train = all_X[:train_count], all_y[:train_count]
    X_val, y_val = all_X[train_count:test_start], all_y[train_count:test_start]
    X_test, y_test = all_X[test_start:], all_y[test_start:]

    # Dates corresponding to test targets (start of each forecast window)
    all_target_dates = df.index[seq_length : seq_length + num_samples]
    test_dates = all_target_dates[test_start:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "target_idx": target_idx,
        "seq_length": seq_length,
        "forecast_horizon": forecast_horizon,
        "test_dates": test_dates,
    }
