"""
Main Pipeline: End-to-End Stock LSTM Training, Feature Comparison,
Graph Visualizations, and Live Prediction.
"""

import os
import argparse
import torch

from data_pipeline import fetch_stock_data, add_technical_indicators, load_custom_dataset
from train_and_compare import (
    plot_feature_correlations,
    plot_market_features,
    run_feature_comparison_experiments,
)
from live_inference import LiveStockPredictor


def main():
    parser = argparse.ArgumentParser(description="Stock LSTM Multi-Feature Training & Live Prediction")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA, SPY)")
    parser.add_argument("--period", type=str, default="2y", help="Historical data window (e.g. 1y, 2y, 5y)")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (e.g. 1d, 1h, 15m, 5m)")
    parser.add_argument("--seq_length", type=int, default=30, help="Sliding window sequence length (timesteps)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per model")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--forecast_horizon", type=int, default=5, help="Number of candles ahead to forecast (e.g. 1, 5, 10, 15)")
    parser.add_argument("--data_path", type=str, default=None, help="Path to your own dataset file (CSV, Excel, etc.) to train on")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save graphs and models")
    args = parser.parse_args()

    # Determine data source and asset identifier
    if args.data_path:
        display_symbol = os.path.splitext(os.path.basename(args.data_path))[0].upper()
        safe_sym = os.path.splitext(os.path.basename(args.data_path))[0].replace("=", "_").replace("-", "_").lower()
        folder_name = f"custom_{safe_sym}_h{args.forecast_horizon}" if args.forecast_horizon > 1 else f"custom_{safe_sym}"
    else:
        display_symbol = args.symbol
        safe_sym = args.symbol.replace("=", "_").replace("-", "_").lower()
        folder_name = f"{safe_sym}_{args.interval}_h{args.forecast_horizon}" if args.forecast_horizon > 1 else f"{safe_sym}_{args.interval}"

    target_output_dir = os.path.join(args.output_dir, folder_name) if args.output_dir == "./outputs" else args.output_dir
    os.makedirs(target_output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Executing pipeline using device: {device.upper()}")
    print(f"[*] Asset / Dataset: {display_symbol}")
    print(f"[*] Forecast Horizon: {args.forecast_horizon} candle(s) ahead")
    print(f"[*] Artifacts will be saved to: {target_output_dir}")

    # -------------------------------------------------------------
    # 1. Fetch Market Data & Compute Technical Indicators
    # -------------------------------------------------------------
    if args.data_path:
        raw_df = load_custom_dataset(args.data_path)
    else:
        raw_df = fetch_stock_data(symbol=args.symbol, period=args.period, interval=args.interval)

    enriched_df = add_technical_indicators(raw_df)

    # -------------------------------------------------------------
    # 2. Generate Initial Feature Analysis Visualizations
    # -------------------------------------------------------------
    corr_features = [
        "Close", "Open", "High", "Low", "Volume",
        "SMA_10", "SMA_20", "SMA_50", "EMA_12",
        "RSI_14", "MACD", "BB_Width", "Volatility_20",
    ]
    corr_path = os.path.join(target_output_dir, "feature_correlations.png")
    plot_feature_correlations(enriched_df, corr_features, save_path=corr_path)

    market_path = os.path.join(target_output_dir, "market_features.png")
    plot_market_features(enriched_df, ticker=display_symbol, save_path=market_path)

    # -------------------------------------------------------------
    # 3. Define Distinct Feature Sets for Comparison
    # -------------------------------------------------------------
    feature_sets = {
        "Univariate (Close Only)": ["Close"],
        "Basic Market (OHLCV)": ["Open", "High", "Low", "Close", "Volume"],
        "Market + Trend (MAs)": [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_10", "SMA_20", "EMA_12",
        ],
        "Full Multi-Feature (Trend + Momentum + Volatility)": [
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "RSI_14", "MACD", "MACD_Signal",
            "BB_Width", "Volatility_20",
        ],
    }

    # -------------------------------------------------------------
    # 4. Train Models & Compare on Graphs
    # -------------------------------------------------------------
    results = run_feature_comparison_experiments(
        df=enriched_df,
        feature_sets=feature_sets,
        target_col="Close",
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        output_dir=target_output_dir,
    )

    # -------------------------------------------------------------
    # 5. Determine Top Performing Feature Set
    # -------------------------------------------------------------
    best_set_name = min(results.keys(), key=lambda k: results[k]["metrics"]["rmse"])
    best_ckpt = results[best_set_name]["ckpt_path"]
    best_rmse = results[best_set_name]["metrics"]["rmse"]
    best_dir_acc = results[best_set_name]["metrics"]["directional_accuracy"]

    print("\n" + "=" * 60)
    print("                EXPERIMENT SUMMARY RESULTS")
    print("=" * 60)
    for name, res in results.items():
        m = res["metrics"]
        print(f"* {name:<45} | RMSE: {m['rmse']:>10.4f} | MAE: {m['mae']:>10.4f} | Dir Acc: {m['directional_accuracy']:>5.1f}%")

    print("-" * 60)
    print(f"[BEST] Feature Set: '{best_set_name}'")
    print(f"       RMSE: {best_rmse:.4f} | Directional Accuracy: {best_dir_acc:.1f}%")
    print(f"       Saved Checkpoint: {best_ckpt}")
    print("=" * 60)

    # -------------------------------------------------------------
    # 6. Run Live Inference with External Data
    # -------------------------------------------------------------
    print("\n[*] Initializing Live Stock Predictor with the Best Model...")
    live_predictor = LiveStockPredictor(checkpoint_path=best_ckpt, device=device)

    # Live prediction using external yfinance feed (or replace with your own API/WebSocket)
    live_forecast = live_predictor.predict_from_live_dataframe(raw_df)

    print("\n" + "#" * 65)
    if args.forecast_horizon > 1:
        print(f"       LIVE MULTI-STEP TRAJECTORY FORECAST FOR {display_symbol}")
        print(f"                   (NEXT {args.forecast_horizon} CANDLES AHEAD)")
    else:
        print(f"          LIVE PREDICTION FOR {display_symbol} (NEXT CANDLE)")
    print("#" * 65)
    print(f"  Reference Bar Timestamp: {live_forecast['timestamp']}")
    print(f"  Current Live Price:      ${live_forecast['current_price']}")

    if args.forecast_horizon > 1:
        print("\n  Forecast Trajectory Breakdown:")
        print(f"  {'Step':<6} | {'Forecast Price':<16} | {'Delta vs Current':<20} | {'Signal'}")
        print("  " + "-" * 58)
        for t in live_forecast["trajectory"]:
            print(f"  t+{t['step']:<3} | ${t['predicted_price']:<15} | {t['delta']:>+10} ({t['pct_change']:>+5.2f}%)   | {t['signal']}")
        print("  " + "-" * 58)
        print(f"  Overall Horizon Trajectory (t+{args.forecast_horizon}): ${live_forecast['terminal_price']} ({live_forecast['terminal_delta']:+} | {live_forecast['terminal_pct_change']:+.2f}%) -> {live_forecast['overall_signal']}")
    else:
        print(f"  Forecasted Next Price:   ${live_forecast['predicted_next_price']}")
        print(f"  Forecasted Delta:        ${live_forecast['predicted_change']:+} ({live_forecast['predicted_pct_change']:+.2f}%)")
        print(f"  Directional Signal:      {live_forecast['signal']}")
    print("#" * 65)
    print(f"\n[OK] Done! Comparison graphs and models are saved in '{target_output_dir}/'.")


if __name__ == "__main__":
    main()
