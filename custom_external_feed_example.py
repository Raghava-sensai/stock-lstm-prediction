"""
Guide & Template: Ingesting Custom External Live Data into the LSTM Predictor.

Use this template to connect:
1. REST APIs (Binance, Alpaca, AlphaVantage, Polygon.io)
2. WebSockets (Live tick streams)
3. Custom CSV / SQL database feeds
"""

import pandas as pd
from live_inference import LiveStockPredictor


def simulate_external_broker_api() -> pd.DataFrame:
    """
    Simulated external feed function.
    Replace this with your actual API call, e.g.:
      requests.get("https://api.yourbroker.com/v1/bars?symbol=AAPL")
    """
    # Sample DataFrame structure expected by the model
    # Note: Needs at least (seq_length + 50) rows to calculate rolling indicators
    import yfinance as yf
    print("[*] Simulating external live stream fetch...")
    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def main():
    # 1. Choose your desired model checkpoint from the comparison experiments
    checkpoint = "./outputs/lstm_basic_market_(ohlcv).pt"
    # Or: "./outputs/lstm_full_multi-feature_(trend_+_momentum_+_volatility).pt"

    predictor = LiveStockPredictor(checkpoint_path=checkpoint)

    # 2. Ingest data from your external source
    external_df = simulate_external_broker_api()

    # 3. Make real-time prediction
    result = predictor.predict_from_live_dataframe(external_df)

    print("\n" + "=" * 55)
    print(f"      EXTERNAL DATA INGESTION FORECAST (Horizon={result['forecast_horizon']} bars)")
    print("=" * 55)
    print(f"Timestamp:       {result['timestamp']}")
    print(f"Latest Price:    ${result['current_price']}")
    
    if result["forecast_horizon"] > 1:
        print("\nTrajectory Breakdown:")
        for t in result["trajectory"]:
            print(f"  Step t+{t['step']}: ${t['predicted_price']} ({t['delta']:+} | {t['pct_change']:+.2f}%) -> {t['signal']}")
        print(f"\nTerminal (Step t+{result['forecast_horizon']}): ${result['terminal_price']} ({result['terminal_delta']:+} | {result['terminal_pct_change']:+.2f}%) -> {result['overall_signal']}")
    else:
        print(f"Predicted Price: ${result['predicted_next_price']}")
        print(f"Predicted Delta: ${result['predicted_change']:+} ({result['predicted_pct_change']:+.2f}%)")
        print(f"Signal:          {result['signal']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
