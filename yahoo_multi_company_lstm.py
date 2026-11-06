"""
Universal Multi-Company Stock LSTM Engine via Yahoo Finance API.
Fetches fresh, split-adjusted, real-time market data directly from Yahoo Finance,
applies Reversible Window-Relative Normalization to eliminate multi-decade distortion,
trains ONE unified PyTorch LSTM across all requested tickers,
and provides an interactive multi-step forecast viewer for any selected company.
"""

import os
import argparse
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lstm_model import StockLSTM
from data_pipeline import add_technical_indicators


PRICE_COLS = ["Open", "High", "Low", "Close", "SMA_10", "SMA_20", "EMA_12"]
OSC_COLS = ["RSI_norm", "MACD_norm", "BB_norm", "Vol_norm", "Volm_norm"]
ALL_FEATURES = PRICE_COLS + OSC_COLS

PRESETS = {
    "nifty_top10": [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS"
    ],
    "us_tech": [
        "NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META"
    ],
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"
    ]
}


# =====================================================================
# 1. YAHOO FINANCE DATA INGESTION & RELATIVE NORMALIZATION
# =====================================================================

def fetch_and_prepare_yahoo_dataset(
    symbols: List[str],
    period: str = "2y",
    interval: str = "1d",
    seq_length: int = 24,
    forecast_horizon: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Downloads fresh market data for multiple tickers via Yahoo Finance API,
    applies Reversible Instance Normalization, and pools sequences into a master dataset.
    """
    print("\n" + "=" * 65)
    print(f"[*] Fetching real-time market data for {len(symbols)} tickers via Yahoo Finance API...")
    print(f"    Period: {period} | Interval: {interval}")
    print("=" * 65)

    close_idx = PRICE_COLS.index("Close")
    company_registry = {}
    all_X_tr, all_y_tr = [], []
    all_X_va, all_y_va = [], []

    for sym in symbols:
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            raw_df = yf.download(sym, period=period, interval=interval, progress=False)
            if isinstance(raw_df.columns, pd.MultiIndex):
                raw_df.columns = raw_df.columns.get_level_values(0)

            raw_df = raw_df.dropna(subset=["Close"])
            if len(raw_df) < (seq_length + forecast_horizon + 30):
                print(f"  [!] Skipping {sym}: insufficient data ({len(raw_df)} rows)")
                continue

            # Standardize index & column types
            raw_df.index = pd.to_datetime(raw_df.index).tz_localize(None)
            for c in ["Open", "High", "Low", "Close", "Volume"]:
                raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

            # Technical indicators
            enriched = add_technical_indicators(raw_df)
            enriched["RSI_norm"] = (enriched["RSI_14"] - 50.0) / 50.0
            enriched["MACD_norm"] = (enriched["MACD"] / (enriched["Close"] + 1e-8)).clip(-0.5, 0.5)
            enriched["BB_norm"] = (enriched["BB_Width"] / 100.0).clip(0.0, 2.0)
            enriched["Vol_norm"] = (enriched["Volatility_20"] / 100.0).clip(0.0, 2.0)
            v_roll = enriched["Volume"].rolling(20, min_periods=1).mean() + 1.0
            enriched["Volm_norm"] = ((enriched["Volume"] - v_roll) / v_roll).clip(-2.0, 5.0)
            enriched = enriched.ffill().bfill()

            P_mat = enriched[PRICE_COLS].values
            O_mat = enriched[OSC_COLS].values
            Close_vec = enriched["Close"].values

            n_seq = len(enriched) - seq_length - forecast_horizon + 1
            if n_seq < 15:
                continue

            # Vectorized sliding window extraction
            P_wins = sliding_window_view(P_mat[:n_seq + seq_length - 1], (seq_length, len(PRICE_COLS)))[:, 0, :, :]
            bases = P_wins[:, -1, close_idx].reshape(-1, 1, 1)

            # Window-Relative scaling
            P_wins_norm = (P_wins - bases) / (bases + 1e-8)
            O_wins = sliding_window_view(O_mat[:n_seq + seq_length - 1], (seq_length, len(OSC_COLS)))[:, 0, :, :]
            X_comp = np.concatenate([P_wins_norm, O_wins], axis=2).astype(np.float32)

            # Target: Multi-step relative percentage return ahead
            future_close = sliding_window_view(Close_vec[seq_length:], forecast_horizon)[:n_seq]
            y_comp = ((future_close - bases[:, 0, :]) / (bases[:, 0, :] + 1e-8)).astype(np.float32)

            # Splits
            n_tr = int(n_seq * train_ratio)
            n_va = max(1, int(n_seq * val_ratio))
            n_te = n_seq - (n_tr + n_va)

            X_tr, y_tr = X_comp[:n_tr], y_comp[:n_tr]
            X_va, y_va = X_comp[n_tr : n_tr + n_va], y_comp[n_tr : n_tr + n_va]
            X_te = X_comp[n_tr + n_va :]

            all_X_tr.append(X_tr)
            all_y_tr.append(y_tr)
            all_X_va.append(X_va)
            all_y_va.append(y_va)

            start_test_idx = seq_length + n_tr + n_va
            test_dates = enriched.index[start_test_idx : start_test_idx + n_te]

            company_registry[sym] = {
                "X_test": X_te,
                "bases_test": bases[n_tr + n_va :, 0, 0],
                "actuals_test": future_close[n_tr + n_va :],
                "test_dates": test_dates,
                "last_raw_df": raw_df.tail(seq_length + 60),
                "last_price": float(Close_vec[-1]),
                "total_records": len(raw_df),
            }
            print(f"  [+] {sym:<14}: {len(raw_df):>4} bars -> {n_tr:>4} train seqs, {n_te:>3} test seqs | Latest: {Close_vec[-1]:.2f}")

        except Exception as e:
            print(f"  [!] Failed {sym}: {e}")

    if not all_X_tr:
        raise ValueError("No tickers were successfully downloaded or processed.")

    X_train_pooled = np.concatenate(all_X_tr, axis=0)
    y_train_pooled = np.concatenate(all_y_tr, axis=0)
    X_val_pooled = np.concatenate(all_X_va, axis=0)
    y_val_pooled = np.concatenate(all_y_va, axis=0)

    print("\n" + "=" * 65)
    print(f"[YAHOO POOLED MASTER DATASET]")
    print(f"  Tickers Successfully Processed:  {len(company_registry)}")
    print(f"  Total Master Training Sequences: {len(X_train_pooled):,}")
    print(f"  Total Master Validation Sequences:{len(X_val_pooled):,}")
    print(f"  Features per Timestep:           {len(ALL_FEATURES)}")
    print(f"  Lookback Window:                 {seq_length} timesteps")
    print(f"  Forecast Horizon Ahead:          {forecast_horizon} candles")
    print("=" * 65)

    return {
        "X_train": X_train_pooled,
        "y_train": y_train_pooled,
        "X_val": X_val_pooled,
        "y_val": y_val_pooled,
        "company_registry": company_registry,
        "seq_length": seq_length,
        "forecast_horizon": forecast_horizon,
    }


# =====================================================================
# 2. UNIVERSAL MODEL TRAINING (ONE MODEL ACROSS ALL YAHOO TICKERS)
# =====================================================================

def train_yahoo_universal_model(
    pooled_data: Dict[str, Any],
    hidden_size: int = 96,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    output_dir: str = "./outputs/yahoo_multi_company",
) -> Dict[str, Any]:
    """
    Trains ONE universal LSTM across all pooled Yahoo Finance tickers using AdamW.
    """
    os.makedirs(output_dir, exist_ok=True)
    seq_length = pooled_data["seq_length"]
    forecast_horizon = pooled_data["forecast_horizon"]
    registry = pooled_data["company_registry"]

    train_tensor_x = torch.tensor(pooled_data["X_train"], dtype=torch.float32)
    train_tensor_y = torch.tensor(pooled_data["y_train"], dtype=torch.float32)
    val_tensor_x = torch.tensor(pooled_data["X_val"], dtype=torch.float32)
    val_tensor_y = torch.tensor(pooled_data["y_val"], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor_x, train_tensor_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor_x, val_tensor_y), batch_size=batch_size, shuffle=False)

    model = StockLSTM(
        input_size=len(ALL_FEATURES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=forecast_horizon,
        dropout=0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print(f"\n[*] Training Universal LSTM on {device.upper()} across all {len(registry)} tickers...")
    best_loss = float("inf")
    best_weights = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_tr = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_tr += loss.item() * len(xb)
        tr_loss = running_tr / len(train_tensor_x)

        model.eval()
        running_va = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running_va += criterion(model(xb), yb).item() * len(xb)
        va_loss = running_va / len(val_tensor_x)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_loss:
            best_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch:>2}/{epochs:>2}] | Train MSE: {tr_loss:.6f} | Val MSE: {va_loss:.6f}")

    if best_weights:
        model.load_state_dict(best_weights)

    # Save model checkpoint
    model_path = os.path.join(output_dir, "yahoo_universal_lstm.pt")
    torch.save({
        "model_state": model.state_dict(),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "seq_length": seq_length,
        "forecast_horizon": forecast_horizon,
    }, model_path)

    # Evaluate each company and save leaderboard
    eval_results = evaluate_all_yahoo_tickers(model, registry, seq_length, forecast_horizon, device, output_dir)

    print("\n" + "=" * 65)
    print(f"[OK] Training & Evaluation Finished!")
    print(f"     Model Saved: {model_path}")
    print(f"     Evaluated Tickers: {len(eval_results)}")
    print("=" * 65)

    return {
        "model": model,
        "model_path": model_path,
        "eval_results": eval_results,
        "registry": registry,
        "history": history,
    }


# =====================================================================
# 3. INDIVIDUAL EVALUATION & RECONSTRUCTION
# =====================================================================

def evaluate_all_yahoo_tickers(
    model: nn.Module,
    registry: Dict[str, Any],
    seq_length: int,
    forecast_horizon: int,
    device: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluates the single model on each ticker's test set and reconstructs actual price paths.
    """
    model.eval()
    eval_results = {}
    summary_rows = []

    for sym, comp in registry.items():
        X_test = comp["X_test"]
        bases_test = comp["bases_test"]
        actuals_test = comp["actuals_test"]

        if len(X_test) == 0:
            continue

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred_rel = model(X_tensor).cpu().numpy()

        # Reconstruct actual prices from relative percentage returns
        predictions = bases_test[:, None] * (1.0 + pred_rel)
        actuals = actuals_test

        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        mae = float(np.mean(np.abs(actuals - predictions)))
        mape = float(np.mean(np.abs((actuals - predictions) / (actuals + 1e-9))) * 100)

        # Directional Hit Rate
        act_dir = (actuals[:, -1] - actuals[:, 0]) > 0
        pred_dir = (predictions[:, -1] - predictions[:, 0]) > 0
        dir_acc = float(np.mean(act_dir == pred_dir) * 100)

        # Real-time latest candle forecast
        last_raw = comp["last_raw_df"]
        enriched = add_technical_indicators(last_raw)
        enriched["RSI_norm"] = (enriched["RSI_14"] - 50.0) / 50.0
        enriched["MACD_norm"] = (enriched["MACD"] / (enriched["Close"] + 1e-8)).clip(-0.5, 0.5)
        enriched["BB_norm"] = (enriched["BB_Width"] / 100.0).clip(0.0, 2.0)
        enriched["Vol_norm"] = (enriched["Volatility_20"] / 100.0).clip(0.0, 2.0)
        v_roll = enriched["Volume"].rolling(20, min_periods=1).mean() + 1.0
        enriched["Volm_norm"] = ((enriched["Volume"] - v_roll) / v_roll).clip(-2.0, 5.0)
        enriched = enriched.ffill().bfill()

        curr_price = float(last_raw["Close"].iloc[-1])
        rec_P = (enriched[PRICE_COLS].iloc[-seq_length:].values - curr_price) / (curr_price + 1e-8)
        rec_O = enriched[OSC_COLS].iloc[-seq_length:].values
        rec_X = np.concatenate([rec_P, rec_O], axis=1)

        in_t = torch.tensor(rec_X, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            latest_pred_rel = model(in_t).cpu().numpy().flatten()

        live_preds = (curr_price * (1.0 + latest_pred_rel)).tolist()
        terminal_pred = live_preds[-1]
        delta = terminal_pred - curr_price
        pct_chg = (delta / curr_price) * 100
        signal = "BULLISH (UP)" if delta > 0 else "BEARISH (DOWN)"

        eval_results[sym] = {
            "ticker": sym,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "directional_accuracy": dir_acc,
            "current_price": curr_price,
            "predicted_trajectory": live_preds,
            "terminal_price": terminal_pred,
            "delta": delta,
            "pct_change": pct_chg,
            "signal": signal,
            "test_dates": comp["test_dates"],
            "actuals": actuals,
            "predictions": predictions,
        }

        summary_rows.append({
            "Ticker": sym,
            "Current Price": round(curr_price, 2),
            "Predicted (t+N)": round(terminal_pred, 2),
            "Expected Delta": f"{delta:+.2f} ({pct_chg:+.2f}%)",
            "Signal": signal,
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "Dir Accuracy (%)": round(dir_acc, 1),
        })

    # Save summary leaderboard
    summary_df = pd.DataFrame(summary_rows).sort_values(by="MAPE (%)")
    summary_path = os.path.join(output_dir, "yahoo_multi_company_leaderboard.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[+] Saved Yahoo Finance multi-company summary table to: {summary_path}")

    # Save evaluation cache
    cache_path = os.path.join(output_dir, "eval_results.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(eval_results, f)
        print(f"[+] Saved evaluation cache for interactive viewer: {cache_path}")
    except Exception as e:
        print(f"[!] Warning: could not cache eval_results: {e}")

    return eval_results


# =====================================================================
# 4. INTERACTIVE FORECAST VIEWER
# =====================================================================

def view_yahoo_forecast(
    ticker: str,
    eval_results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Selects any Yahoo Finance ticker and displays its forecasted multi-step trajectory,
    historical test predictions vs actuals, and key performance metrics.
    """
    ticker = ticker.upper()
    if ticker not in eval_results:
        available = list(eval_results.keys())[:10]
        raise KeyError(f"Ticker '{ticker}' not found. Available: {available}...")

    res = eval_results[ticker]
    horizon = len(res["predicted_trajectory"])

    print("\n" + "=" * 65)
    print(f"       UNIVERSAL LSTM FORECAST FOR: {ticker}")
    print("=" * 65)
    print(f"  Current Price:           {res['current_price']:.2f}")
    print(f"  Predicted Net Target:    {res['terminal_price']:.2f} ({res['delta']:+.2f} | {res['pct_change']:+.2f}%)")
    print(f"  Directional Signal:      {res['signal']}")
    print(f"  Model Test RMSE:         {res['rmse']:.2f}")
    print(f"  Model Test MAPE:         {res['mape']:.2f}%")
    print(f"  Directional Accuracy:    {res['directional_accuracy']:.1f}%")
    print("-" * 65)
    print("  Forecasted Step-by-Step Trajectory:")
    for step_idx, p in enumerate(res["predicted_trajectory"], start=1):
        d = p - res["current_price"]
        pct = (d / res["current_price"]) * 100
        sig = "UP" if d > 0 else "DOWN"
        print(f"    t+{step_idx:<2} -> {p:<8.2f} (Delta: {d:>+7.2f} | {pct:>+5.2f}%) [{sig}]")
    print("=" * 65)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Subplot 1: Test Actual vs Predicted
    act_step1 = res["actuals"][:, 0]
    pred_step1 = res["predictions"][:, 0]
    test_dates = res["test_dates"][:len(act_step1)]

    axes[0].plot(test_dates, act_step1, label="Actual Price", color="black", linewidth=2)
    axes[0].plot(test_dates, pred_step1, label=f"Universal LSTM (RMSE: {res['rmse']:.2f}, MAPE: {res['mape']:.2f}%)", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"{ticker}: Test Prediction vs Ground Truth", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Projected Forecast Cone
    steps = [0] + list(range(1, horizon + 1))
    prices = [res["current_price"]] + res["predicted_trajectory"]
    color_sig = "#2ca02c" if res["delta"] > 0 else "#d62728"

    axes[1].plot(steps, prices, marker="o", linewidth=2.5, color=color_sig, label=f"Forecast Path ({res['signal']})")
    axes[1].axhline(res["current_price"], color="gray", linestyle=":", label="Current Reference Price")
    axes[1].set_title(f"{ticker}: Projected {horizon}-Candle Trajectory Ahead", fontweight="bold")
    axes[1].set_xlabel("Horizon Step (t + k)")
    axes[1].set_ylabel("Price")
    axes[1].set_xticks(steps)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"[+] Saved {ticker} visualization to: {save_path}")
    if show_plot:
        plt.show()
    plt.close()


def launch_interactive_widget(eval_results: Dict[str, Any]):
    """
    Interactive dropdown for Jupyter / Google Colab to switch between tickers instantly.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        tickers = sorted(list(eval_results.keys()))
        if not tickers:
            print("[!] No tickers available in evaluation results.")
            return

        dropdown = widgets.Dropdown(
            options=tickers,
            value=tickers[0],
            description="Select Ticker:",
            layout=widgets.Layout(width="380px")
        )
        out = widgets.Output()

        def on_value_change(change):
            if change["name"] == "value":
                with out:
                    clear_output(wait=True)
                    view_yahoo_forecast(change["new"], eval_results, show_plot=True)

        dropdown.observe(on_value_change, names="value")
        display(dropdown)
        with out:
            view_yahoo_forecast(tickers[0], eval_results, show_plot=True)
        display(out)
    except ImportError:
        print("[!] ipywidgets or IPython not installed. Use view_yahoo_forecast('TICKER', eval_results) directly.")


# =====================================================================
# 5. CLI RUNNER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Universal Yahoo Finance Multi-Company LSTM")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated stock symbols (e.g. 'RELIANCE.NS,TCS.NS,AAPL,NVDA')")
    parser.add_argument("--preset", type=str, default="nifty_top10", choices=["nifty_top10", "us_tech", "crypto", "custom"], help="Preset ticker list")
    parser.add_argument("--period", type=str, default="2y", help="Historical period (e.g. 1y, 2y, 5y)")
    parser.add_argument("--interval", type=str, default="1d", help="Candle interval (e.g. 1d, 1h)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for master pooled data")
    parser.add_argument("--forecast_horizon", type=int, default=5, help="Candles ahead to predict")
    parser.add_argument("--seq_length", type=int, default=24, help="Lookback window")
    parser.add_argument("--select_ticker", type=str, default="", help="Ticker to plot after training")
    parser.add_argument("--output_dir", type=str, default="./outputs/yahoo_multi_company", help="Output directory")
    parser.add_argument("--view_only", action="store_true", help="Skip training and view forecast from cached evaluation")
    args = parser.parse_args()

    # View only mode from cache
    if args.view_only:
        cache_path = os.path.join(args.output_dir, "eval_results.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file '{cache_path}' not found. Please train first.")
        with open(cache_path, "rb") as f:
            eval_results = pickle.load(f)
        target = args.select_ticker.upper() if args.select_ticker else list(eval_results.keys())[0]
        chart_file = os.path.join(args.output_dir, f"{target}_forecast.png")
        view_yahoo_forecast(target, eval_results, save_path=chart_file, show_plot=False)
        return

    # Parse symbols
    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = PRESETS.get(args.preset, PRESETS["nifty_top10"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Universal Yahoo Finance LSTM starting on device: {device.upper()}")

    # 1. Download and prepare data
    pooled_data = fetch_and_prepare_yahoo_dataset(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
    )

    # 2. Train universal model
    summary = train_yahoo_universal_model(
        pooled_data=pooled_data,
        hidden_size=96,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
    )

    # 3. View selected ticker
    eval_results = summary["eval_results"]
    target = args.select_ticker.upper() if args.select_ticker in eval_results else list(eval_results.keys())[0]
    chart_file = os.path.join(args.output_dir, f"{target}_forecast.png")
    view_yahoo_forecast(target, eval_results, save_path=chart_file, show_plot=False)


if __name__ == "__main__":
    main()
