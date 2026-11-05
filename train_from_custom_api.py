"""
Phase 1 Model Training Using Custom External API Data.

Use this script to:
1. Ingest historical or streaming bar data from your own API (REST / JSON / Broker / CSV).
2. Clean and standardize the API response into OHLCV DataFrame format.
3. Train Phase 1 LSTM models across multiple feature sets.
4. Output comparison charts and save the top-performing model checkpoint.
"""

import os
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
import torch

from data_pipeline import add_technical_indicators
from train_and_compare import (
    plot_feature_correlations,
    plot_market_features,
    run_feature_comparison_experiments,
    format_price,
)
from live_inference import LiveStockPredictor


# =====================================================================
# 1. YOUR API INGESTION SPACE (Plug Your API Details Here)
# =====================================================================

def fetch_data_from_your_api(
    symbol: str = "BTCUSDT",
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    TEMPLATE: Fetch market data from your external API.
    
    Supported out-of-the-box examples:
    - Custom REST API endpoint (requests.get / requests.post)
    - Binance, Alpaca, Polygon, AlphaVantage, or your proprietary server
    """
    print(f"[*] Ingesting data for {symbol} from custom API...")

    # --- Option A: Real Custom REST API ---
    if api_url:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        params = {"symbol": symbol, "limit": 1000}
        
        response = requests.get(api_url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        raw_json = response.json()
        
        # Parse payload into DataFrame
        df = parse_api_payload(raw_json)
        return df

    # --- Option B: Default Fallback Demo (Simulated Broker API Feed) ---
    print("    [Info] No custom api_url passed. Using simulated live market data feed for demonstration.")
    import yfinance as yf
    raw = yf.download(symbol if "-" in symbol or "=" in symbol else f"{symbol[:3]}-{symbol[3:]}", period="60d", interval="1h", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw.dropna()


def parse_api_payload(raw_json: Any) -> pd.DataFrame:
    """
    Standardizes any JSON API format into the expected DataFrame structure:
    Required Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    Index: DatetimeIndex (sorted ascending)
    """
    # Example 1: If API returns a list of dictionaries:
    # [{"timestamp": "...", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 15000}, ...]
    if isinstance(raw_json, list) and isinstance(raw_json[0], dict):
        df = pd.DataFrame(raw_json)
        
        # Standardize column naming
        rename_map = {
            "o": "Open", "open": "Open",
            "h": "High", "high": "High",
            "l": "Low", "low": "Low",
            "c": "Close", "close": "Close",
            "v": "Volume", "vol": "Volume", "volume": "Volume",
            "t": "Datetime", "time": "Datetime", "timestamp": "Datetime", "date": "Datetime",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")

    # Example 2: If API returns Binance-style kline arrays:
    # [[1499040000000, "0.01634790", "0.80000000", "0.01575800", "0.01577100", "148976.11427815", ...], ...]
    elif isinstance(raw_json, list) and isinstance(raw_json[0], list):
        df = pd.DataFrame(raw_json).iloc[:, :6]
        df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
        df = df.set_index("Datetime")

    else:
        df = pd.DataFrame(raw_json)

    # Ensure float types
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index().dropna()
    return df


# =====================================================================
# 2. PHASE 1 TRAINING ENGINE (Runs Automatically on Your API Data)
# =====================================================================

def train_phase1_from_api(
    df: pd.DataFrame,
    symbol: str = "CUSTOM_ASSET",
    seq_length: int = 24,
    forecast_horizon: int = 5,
    epochs: int = 15,
    batch_size: int = 32,
    output_dir: str = "./outputs/custom_api_phase1",
) -> Dict[str, Any]:
    """
    Executes complete Phase 1 Training on data supplied from your API:
    1. Computes technical indicators.
    2. Trains and compares 4 feature sets.
    3. Generates performance and trajectory graphs.
    4. Saves model checkpoints for live Phase 2 inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "=" * 65)
    print(f"        PHASE 1 MODEL TRAINING (FROM CUSTOM API DATA)")
    print("=" * 65)
    print(f"  Asset / Symbol:     {symbol}")
    print(f"  Data Records:       {len(df)} candles ({df.index[0]} to {df.index[-1]})")
    print(f"  Input Window:       {seq_length} timesteps")
    print(f"  Forecast Horizon:   {forecast_horizon} candle(s) ahead")
    print(f"  Device:             {device.upper()}")
    print(f"  Output Directory:   {output_dir}")
    print("=" * 65)

    # 1. Feature Engineering
    enriched_df = add_technical_indicators(df)

    # 2. Plot Exploratory Graphs
    corr_features = [
        "Close", "Open", "High", "Low", "Volume",
        "SMA_10", "SMA_20", "SMA_50", "EMA_12",
        "RSI_14", "MACD", "BB_Width", "Volatility_20",
    ]
    plot_feature_correlations(enriched_df, corr_features, save_path=os.path.join(output_dir, "feature_correlations.png"))
    plot_market_features(enriched_df, ticker=symbol, save_path=os.path.join(output_dir, "market_features.png"))

    # 3. Define 4 Distinct Feature Sets for Comparison
    feature_sets = {
        "Univariate (Close Only)": ["Close"],
        "Basic Market (OHLCV)": ["Open", "High", "Low", "Close", "Volume"],
        "Market + Trend (MAs)": [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_10", "SMA_20", "EMA_12",
        ],
        "Full Multi-Feature": [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "RSI_14", "MACD", "MACD_Signal",
            "BB_Width", "Volatility_20",
        ],
    }

    # 4. Train Models & Compare
    results = run_feature_comparison_experiments(
        df=enriched_df,
        feature_sets=feature_sets,
        target_col="Close",
        seq_length=seq_length,
        forecast_horizon=forecast_horizon,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
    )

    # 5. Best Model Identification
    best_name = min(results.keys(), key=lambda k: results[k]["metrics"]["rmse"])
    best_ckpt = results[best_name]["ckpt_path"]
    best_rmse = results[best_name]["metrics"]["rmse"]
    best_dir = results[best_name]["metrics"]["directional_accuracy"]

    print("\n" + "=" * 65)
    print("                PHASE 1 SUMMARY REPORT")
    print("=" * 65)
    for name, res in results.items():
        m = res["metrics"]
        print(f"* {name:<25} | RMSE: {format_price(m['rmse']):>10} | MAE: {format_price(m['mae']):>10} | Dir Acc: {m['directional_accuracy']:>5.1f}%")
    
    print("-" * 65)
    print(f"[BEST] Top Architecture: '{best_name}'")
    print(f"       Overall RMSE: {format_price(best_rmse)} | Directional Hit Rate: {best_dir:.1f}%")
    print(f"       Saved Model:  {best_ckpt}")
    print("=" * 65)

    return {
        "best_model_name": best_name,
        "best_checkpoint": best_ckpt,
        "results": results,
        "output_dir": output_dir,
    }


# =====================================================================
# 3. MAIN RUNNER
# =====================================================================

def main():
    # --- Plug your API configuration here ---
    MY_API_URL = None         # e.g., "https://api.binance.com/api/v3/klines" or your custom server
    MY_API_KEY = None         # e.g., "your_secret_api_key_here"
    SYMBOL = "BTC-USD"        # e.g., "BTCUSDT", "ETHUSDT", "EURUSD", "AAPL"
    
    # 1. Fetch live/historical data directly from your API
    api_data = fetch_data_from_your_api(symbol=SYMBOL, api_url=MY_API_URL, api_key=MY_API_KEY)

    # 2. Train Phase 1 models using the API data
    phase1_summary = train_phase1_from_api(
        df=api_data,
        symbol=SYMBOL,
        seq_length=24,        # Lookback sequence length (candles)
        forecast_horizon=5,   # Multi-step ahead horizon (candles)
        epochs=15,            # Training epochs
        output_dir="./outputs/custom_api_phase1",
    )

    # 3. Test immediate live forecast with the trained model
    best_checkpoint = phase1_summary["best_checkpoint"]
    predictor = LiveStockPredictor(checkpoint_path=best_checkpoint)
    live_pred = predictor.predict_from_live_dataframe(api_data)

    print("\n" + "#" * 65)
    print("      SAMPLE REAL-TIME FORECAST FROM TRAINED API MODEL")
    print("#" * 65)
    print(f"  Reference Bar:  {live_pred['timestamp']}")
    print(f"  Current Price:  ${live_pred['current_price']}")
    print(f"  Forecast (t+1): ${live_pred['predicted_next_price']} ({live_pred['predicted_change']:+} | {live_pred['predicted_pct_change']:+.2f}%) -> {live_pred['signal']}")
    print(f"  Terminal (t+{live_pred['forecast_horizon']}): ${live_pred['terminal_price']} ({live_pred['terminal_delta']:+} | {live_pred['terminal_pct_change']:+.2f}%) -> {live_pred['overall_signal']}")
    print("#" * 65)


if __name__ == "__main__":
    main()
