"""
Live Data Ingestion and Real-Time Inference Module.
Loads trained LSTM checkpoints and processes live feeds from external sources.
"""

from typing import Dict, Any, Optional
import time
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from lstm_model import StockLSTM
from data_pipeline import add_technical_indicators


class LiveStockPredictor:
    """
    Handles live data ingestion, feature generation on the fly,
    and real-time next-tick/candle inference using a trained LSTM.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        print(f"[*] Loading trained model checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        self.feature_cols = ckpt["feature_cols"]
        self.feature_scaler = ckpt["feature_scaler"]
        self.target_scaler = ckpt["target_scaler"]
        self.seq_length = ckpt["seq_length"]
        self.forecast_horizon = ckpt.get("forecast_horizon", 1)
        self.hidden_size = ckpt["hidden_size"]

        # Instantiate model architecture
        self.model = StockLSTM(
            input_size=len(self.feature_cols),
            hidden_size=self.hidden_size,
            num_layers=2,
            output_size=self.forecast_horizon,
            dropout=0.0,  # Inference mode
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device)
        self.model.eval()
        print(f"[+] Loaded model successfully (Horizon={self.forecast_horizon} candles). Expects {len(self.feature_cols)} features: {self.feature_cols}")

    def predict_from_live_dataframe(self, live_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Receives a DataFrame containing recent live market data (OHLCV),
        computes required technical indicators, extracts the latest sequence window,
        and predicts the next N time steps' Close price trajectory.
        """
        # Adaptively determine warm-up requirement based on features
        warmup = 0
        if any("SMA_50" in f or "EMA_50" in f for f in self.feature_cols):
            warmup = 55
        elif any("SMA_20" in f or "BB_" in f or "Volatility" in f for f in self.feature_cols):
            warmup = 25
        elif any("RSI_14" in f or "SMA_10" in f or "MACD" in f for f in self.feature_cols):
            warmup = 20

        min_required = self.seq_length + warmup
        if len(live_df) < min_required:
            raise ValueError(
                f"live_df must contain at least {min_required} rows (sequence length {self.seq_length} + {warmup} warmup periods for indicators). Provided: {len(live_df)}"
            )

        # 1. Compute technical indicators if needed
        if warmup > 0:
            processed_df = add_technical_indicators(live_df)
        else:
            processed_df = live_df.copy()

        # Check all required features exist
        missing = [f for f in self.feature_cols if f not in processed_df.columns]
        if missing:
            raise KeyError(f"Missing required features in live data: {missing}")

        # 2. Slice the most recent sequence window
        recent_features = processed_df[self.feature_cols].iloc[-self.seq_length :].values

        # 3. Scale using fitted training scaler
        scaled_features = self.feature_scaler.transform(recent_features)

        # Reshape to (1, seq_length, num_features)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. Predict
        with torch.no_grad():
            scaled_pred = self.model(input_tensor).cpu().numpy()

        # 5. Inverse transform predictions across all horizon steps
        predicted_prices = self.target_scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten().tolist()
        current_price = float(live_df["Close"].iloc[-1])
        latest_timestamp = live_df.index[-1]

        trajectory = []
        for step_idx, p in enumerate(predicted_prices, start=1):
            step_delta = p - current_price
            step_pct = (step_delta / current_price) * 100
            step_signal = "BULLISH (UP)" if step_delta > 0 else "BEARISH (DOWN)"
            trajectory.append({
                "step": step_idx,
                "predicted_price": round(p, 4 if p < 5 else 2),
                "delta": round(step_delta, 4 if p < 5 else 2),
                "pct_change": round(step_pct, 2),
                "signal": step_signal,
            })

        immediate_pred = trajectory[0]["predicted_price"]
        immediate_delta = trajectory[0]["delta"]
        immediate_pct = trajectory[0]["pct_change"]
        immediate_signal = trajectory[0]["signal"]

        terminal_pred = trajectory[-1]["predicted_price"]
        terminal_delta = trajectory[-1]["delta"]
        terminal_pct = trajectory[-1]["pct_change"]
        overall_signal = trajectory[-1]["signal"]

        return {
            "timestamp": str(latest_timestamp),
            "current_price": round(current_price, 4 if current_price < 5 else 2),
            "forecast_horizon": self.forecast_horizon,
            "predicted_next_price": immediate_pred,
            "predicted_change": immediate_delta,
            "predicted_pct_change": immediate_pct,
            "signal": immediate_signal,
            "terminal_price": terminal_pred,
            "terminal_delta": terminal_delta,
            "terminal_pct_change": terminal_pct,
            "overall_signal": overall_signal,
            "trajectory": trajectory,
        }

    def fetch_and_predict_live(
        self,
        symbol: str,
        interval: str = "1m",
        source: str = "yfinance",
    ) -> Dict[str, Any]:
        """
        Pull real-time live data from external source and make immediate forecast.
        """
        print(f"[*] Ingesting live data for {symbol} ({interval} interval) via {source}...")
        if source == "yfinance":
            # For 1m/5m interval, fetch last 5 days of intraday data to have enough indicator warm-up
            period = "5d" if interval in ["1m", "2m", "5m"] else "60d"
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise RuntimeError(f"Could not fetch live data for {symbol}.")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
        else:
            raise NotImplementedError(f"External source '{source}' not configured. Pass live DataFrame directly.")

        return self.predict_from_live_dataframe(df)


def run_live_feed_monitor(
    checkpoint_path: str,
    symbol: str = "AAPL",
    interval: str = "1m",
    iterations: int = 5,
    sleep_seconds: int = 15,
):
    """
    Simulate or run a live feed monitor that queries external data periodically,
    forecasts the next tick/candle, and prints signals.
    """
    predictor = LiveStockPredictor(checkpoint_path)

    print(f"\n=======================================================")
    print(f"[*] Starting Live Prediction Feed for {symbol} ({interval})")
    print(f"=======================================================")

    for i in range(1, iterations + 1):
        try:
            res = predictor.fetch_and_predict_live(symbol=symbol, interval=interval)
            print(f"\n[Tick {i}/{iterations}] Time: {res['timestamp']}")
            print(f"  Current Live Price:    ${res['current_price']:.2f}")
            print(f"  Predicted Next Price:  ${res['predicted_next_price']:.2f}")
            print(f"  Expected Delta:        ${res['predicted_change']:+.2f} ({res['predicted_pct_change']:+.2f}%)")
            print(f"  Forecast Signal:       {res['signal']}")
        except Exception as e:
            print(f"[!] Error reading live tick: {e}")

        if i < iterations:
            print(f"[*] Waiting {sleep_seconds}s for next live tick...")
            time.sleep(sleep_seconds)
