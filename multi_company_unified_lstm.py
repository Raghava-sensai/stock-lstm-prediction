"""
Universal Multi-Company Stock LSTM Model.
Trains ONE unified, generalized LSTM model across an entire folder of company CSV files.
Allows selecting or clicking any company stock to view its specific multi-step forecast and graphs.
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from lstm_model import StockLSTM
from data_pipeline import add_technical_indicators, create_sliding_sequences
from train_and_compare import format_price


# =====================================================================
# 1. BATCH DATASET LOADER FOR MULTIPLE COMPANIES
# =====================================================================

def standardize_company_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names and index for any company CSV."""
    df.columns = [str(c).strip() for c in df.columns]
    date_candidates = ["date", "datetime", "timestamp", "time", "date_time", "dt"]
    matched = None
    for col in df.columns:
        if col.lower() in date_candidates:
            matched = col
            break

    if matched:
        df["Datetime"] = pd.to_datetime(df[matched], errors="coerce")
        df = df.set_index("Datetime")
        if matched != "Datetime" and matched in df.columns:
            df = df.drop(columns=[matched])
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            df.index = pd.date_range(start="2020-01-01", periods=len(df), freq="D")

    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ["open", "o"]: col_map[col] = "Open"
        elif cl in ["high", "h"]: col_map[col] = "High"
        elif cl in ["low", "l"]: col_map[col] = "Low"
        elif cl in ["close", "c", "price", "last", "settle", "adj close", "adj_close"]:
            if "Close" not in col_map.values(): col_map[col] = "Close"
        elif cl in ["volume", "vol", "v", "qty", "quantity"]: col_map[col] = "Volume"

    df = df.rename(columns=col_map)
    if "Close" not in df.columns:
        nums = df.select_dtypes(include=[np.number]).columns
        if len(nums) > 0: df["Close"] = df[nums[0]]
        else: raise ValueError("No valid Close price column found.")

    if "Open" not in df.columns: df["Open"] = df["Close"]
    if "High" not in df.columns: df["High"] = df["Close"]
    if "Low" not in df.columns: df["Low"] = df["Close"]
    if "Volume" not in df.columns: df["Volume"] = 1.0

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_index().dropna(subset=["Close"])


def prepare_multi_company_dataset(
    folder_path: str,
    feature_cols: List[str],
    seq_length: int = 24,
    forecast_horizon: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_companies: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Ingests all company CSV files in a folder, scales each company independently to prevent
    leakage/cross-price distortion, and aggregates sequences into ONE master training dataset.
    """
    if os.path.isfile(folder_path) and folder_path.lower().endswith(".csv"):
        csv_files = [folder_path]
    else:
        found_files = set(glob.glob(os.path.join(folder_path, "*.csv")))
        found_files.update(glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True))
        csv_files = sorted(list(found_files))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in target: '{folder_path}'")

    if max_companies:
        csv_files = csv_files[:max_companies]

    print(f"[*] Found {len(csv_files)} company CSV files to ingest into ONE universal model.")

    target_idx = feature_cols.index("Close")
    company_registry = {}
    
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    all_X_test, all_y_test = [], []

    for file_path in csv_files:
        ticker = os.path.splitext(os.path.basename(file_path))[0].upper()
        try:
            raw_df = pd.read_csv(file_path)
            clean_df = standardize_company_df(raw_df)
            if len(clean_df) < (seq_length + forecast_horizon + 50):
                continue  # Skip files with too few records for rolling indicators

            enriched_df = add_technical_indicators(clean_df)
            raw_matrix = enriched_df[feature_cols].values
            n = len(raw_matrix)
            train_end = int(n * train_ratio)

            # Fit scaler ONLY on company's own training slice
            f_scaler = MinMaxScaler(feature_range=(0, 1))
            f_scaler.fit(raw_matrix[:train_end])
            scaled_matrix = f_scaler.transform(raw_matrix)

            t_scaler = MinMaxScaler(feature_range=(0, 1))
            t_scaler.min_ = np.array([f_scaler.min_[target_idx]])
            t_scaler.scale_ = np.array([f_scaler.scale_[target_idx]])

            # Build sliding sequences
            X_comp, y_comp = create_sliding_sequences(scaled_matrix, target_idx, seq_length, forecast_horizon)
            n_seq = len(X_comp)
            if n_seq < 10:
                continue

            n_tr = int(n_seq * train_ratio)
            n_va = max(1, int(n_seq * val_ratio))
            n_te = n_seq - (n_tr + n_va)

            X_tr, y_tr = X_comp[:n_tr], y_comp[:n_tr]
            X_va, y_va = X_comp[n_tr : n_tr + n_va], y_comp[n_tr : n_tr + n_va]
            X_te, y_te = X_comp[n_tr + n_va :], y_comp[n_tr + n_va :]

            all_X_train.append(X_tr)
            all_y_train.append(y_tr)
            all_X_val.append(X_va)
            all_y_val.append(y_va)
            all_X_test.append(X_te)
            all_y_test.append(y_te)

            start_test_idx = seq_length + n_tr + n_va
            test_dates = enriched_df.index[start_test_idx : start_test_idx + n_te]

            # Save company metadata for individual inference
            company_registry[ticker] = {
                "feature_scaler": f_scaler,
                "target_scaler": t_scaler,
                "last_raw_df": clean_df.tail(seq_length + 60),
                "X_test": X_te,
                "y_test": y_te,
                "test_dates": test_dates,
                "last_price": float(clean_df["Close"].iloc[-1]),
                "total_records": len(clean_df),
            }
            print(f"  [+] {ticker:<15}: {len(clean_df):>5} candles -> {n_tr:>5} train seqs, {n_te:>4} test seqs")

        except Exception as e:
            print(f"  [!] Skipped {ticker}: {e}")

    # Pool all company data into single tensors
    X_train_pooled = np.concatenate(all_X_train, axis=0)
    y_train_pooled = np.concatenate(all_y_train, axis=0)
    X_val_pooled = np.concatenate(all_X_val, axis=0)
    y_val_pooled = np.concatenate(all_y_val, axis=0)

    print("\n" + "=" * 65)
    print(f"[POOLED DATASET SUMMARY]")
    print(f"  Companies Successfully Processed: {len(company_registry)}")
    print(f"  Total Master Training Sequences:  {len(X_train_pooled):,}")
    print(f"  Total Master Validation Sequences:{len(X_val_pooled):,}")
    print(f"  Feature Dimensions:               {len(feature_cols)} features")
    print(f"  Lookback Window:                  {seq_length} timesteps")
    print(f"  Forecast Horizon:                 {forecast_horizon} candles ahead")
    print("=" * 65)

    return {
        "X_train": X_train_pooled,
        "y_train": y_train_pooled,
        "X_val": X_val_pooled,
        "y_val": y_val_pooled,
        "company_registry": company_registry,
        "feature_cols": feature_cols,
        "seq_length": seq_length,
        "forecast_horizon": forecast_horizon,
    }


# =====================================================================
# 2. UNIVERSAL MODEL TRAINING (ONE MODEL FOR ALL COMPANIES)
# =====================================================================

def train_universal_model(
    pooled_data: Dict[str, Any],
    hidden_size: int = 96,
    num_layers: int = 2,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    output_dir: str = "./outputs/universal_multi_company",
) -> Dict[str, Any]:
    """
    Trains ONE universal LSTM network across all pooled companies using AdamW.
    """
    os.makedirs(output_dir, exist_ok=True)
    feature_cols = pooled_data["feature_cols"]
    forecast_horizon = pooled_data["forecast_horizon"]
    seq_length = pooled_data["seq_length"]
    registry = pooled_data["company_registry"]

    train_tensor_x = torch.tensor(pooled_data["X_train"], dtype=torch.float32)
    train_tensor_y = torch.tensor(pooled_data["y_train"], dtype=torch.float32)
    val_tensor_x = torch.tensor(pooled_data["X_val"], dtype=torch.float32)
    val_tensor_y = torch.tensor(pooled_data["y_val"], dtype=torch.float32)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y),
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize single universal model
    model = StockLSTM(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=forecast_horizon,
        dropout=0.2,
    )
    model.to(device)

    criterion = nn.MSELoss()
    # Decoupled AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    print(f"\n[*] Training Universal LSTM on {device.upper()} across all {len(registry)} companies...")
    best_loss = float("inf")
    best_weights = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_train += loss.item() * len(X_b)

        tr_loss = running_train / len(train_tensor_x)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                running_val += loss.item() * len(X_b)

        va_loss = running_val / len(val_tensor_x)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_loss:
            best_loss = va_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch [{epoch:>2}/{epochs:>2}] | Train MSE: {tr_loss:.6f} | Val MSE: {va_loss:.6f}")

    if best_weights:
        model.load_state_dict(best_weights)

    # Save universal model checkpoint
    model_path = os.path.join(output_dir, "universal_lstm_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "feature_cols": feature_cols,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "seq_length": seq_length,
        "forecast_horizon": forecast_horizon,
    }, model_path)

    # Evaluate each company individually using this single universal model
    company_evaluations = evaluate_all_companies(model, registry, feature_cols, seq_length, forecast_horizon, device, output_dir)

    print("\n" + "=" * 65)
    print(f"[OK] Universal Model Training Complete!")
    print(f"    Saved Model Checkpoint: {model_path}")
    print(f"    Total Companies Evaluated: {len(company_evaluations)}")
    print("=" * 65)

    return {
        "model": model,
        "model_path": model_path,
        "company_evaluations": company_evaluations,
        "registry": registry,
        "history": history,
    }


# =====================================================================
# 3. INDIVIDUAL COMPANY EVALUATION WITH THE SINGLE MODEL
# =====================================================================

def evaluate_all_companies(
    model: nn.Module,
    registry: Dict[str, Any],
    feature_cols: List[str],
    seq_length: int,
    forecast_horizon: int,
    device: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Runs the single universal model on each company's specific test set and collects metrics.
    """
    model.eval()
    eval_results = {}
    summary_rows = []

    for ticker, comp in registry.items():
        X_test = comp["X_test"]
        y_test = comp["y_test"]
        t_scaler = comp["target_scaler"]

        if len(X_test) == 0:
            continue

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            scaled_preds = model(X_tensor).cpu().numpy()

        actuals = t_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, forecast_horizon)
        predictions = t_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).reshape(-1, forecast_horizon)

        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        mae = float(np.mean(np.abs(actuals - predictions)))
        mape = float(np.mean(np.abs((actuals - predictions) / (actuals + 1e-9))) * 100)

        # Directional Accuracy (t+0 to t+N)
        act_dir = (actuals[:, -1] - actuals[:, 0]) > 0
        pred_dir = (predictions[:, -1] - predictions[:, 0]) > 0
        dir_acc = float(np.mean(act_dir == pred_dir) * 100)

        # Real-time latest candle forecast from this company's most recent sequence
        last_raw = comp["last_raw_df"]
        enriched = add_technical_indicators(last_raw)
        recent_feats = enriched[feature_cols].iloc[-seq_length:].values
        scaled_recent = comp["feature_scaler"].transform(recent_feats)
        in_t = torch.tensor(scaled_recent, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            latest_pred_scaled = model(in_t).cpu().numpy()

        live_preds = t_scaler.inverse_transform(latest_pred_scaled.reshape(-1, 1)).flatten().tolist()
        curr_price = float(last_raw["Close"].iloc[-1])
        terminal_pred = live_preds[-1]
        delta = terminal_pred - curr_price
        pct_chg = (delta / curr_price) * 100
        signal = "BULLISH (UP)" if delta > 0 else "BEARISH (DOWN)"

        eval_results[ticker] = {
            "ticker": ticker,
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
            "Ticker": ticker,
            "Current Price": round(curr_price, 2),
            "Predicted (t+N)": round(terminal_pred, 2),
            "Expected Delta": f"{delta:+.2f} ({pct_chg:+.2f}%)",
            "Signal": signal,
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "MAPE (%)": round(mape, 2),
            "Dir Accuracy (%)": round(dir_acc, 1),
        })

    # Save summary leaderboard
    summary_df = pd.DataFrame(summary_rows).sort_values(by="MAPE (%)")
    summary_path = os.path.join(output_dir, "multi_company_leaderboard.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[+] Saved multi-company summary table to: {summary_path}")

    # Save evaluation cache for instant interactive selection
    import pickle
    cache_path = os.path.join(output_dir, "eval_results.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(eval_results, f)
        print(f"[+] Saved evaluation cache for interactive viewer: {cache_path}")
    except Exception as e:
        print(f"[!] Warning: could not cache eval_results: {e}")

    return eval_results


# =====================================================================
# 4. INTERACTIVE COMPANY VIEWER (Click / Select Any Company)
# =====================================================================

def view_company_forecast(
    ticker: str,
    eval_results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Selects any company stock and displays its forecasted multi-step trajectory,
    historical test predictions vs actuals, and key performance metrics.
    """
    ticker = ticker.upper()
    if ticker not in eval_results:
        available = list(eval_results.keys())[:10]
        raise KeyError(f"Ticker '{ticker}' not found. Available examples: {available}...")

    res = eval_results[ticker]
    horizon = len(res["predicted_trajectory"])

    print("\n" + "=" * 65)
    print(f"       UNIVERSAL LSTM FORECAST FOR: {ticker}")
    print("=" * 65)
    print(f"  Current Stock Price:     ${res['current_price']:.2f}")
    print(f"  Predicted Net Target:    ${res['terminal_price']:.2f} ({res['delta']:+.2f} | {res['pct_change']:+.2f}%)")
    print(f"  Directional Signal:      {res['signal']}")
    print(f"  Model Test RMSE:         ${res['rmse']:.2f}")
    print(f"  Model Test MAPE:         {res['mape']:.2f}%")
    print(f"  Directional Accuracy:    {res['directional_accuracy']:.1f}%")
    print("-" * 65)
    print("  Forecasted Step-by-Step Trajectory:")
    for step_idx, p in enumerate(res["predicted_trajectory"], start=1):
        d = p - res["current_price"]
        pct = (d / res["current_price"]) * 100
        sig = "UP" if d > 0 else "DOWN"
        print(f"    t+{step_idx:<2} -> ${p:<8.2f} (Delta: {d:>+7.2f} | {pct:>+5.2f}%) [{sig}]")
    print("=" * 65)

    # Plot Visual
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Subplot 1: Test Set Ground Truth vs Predicted
    act_step1 = res["actuals"][:, 0]
    pred_step1 = res["predictions"][:, 0]
    test_dates = res["test_dates"][:len(act_step1)]
    axes[0].plot(test_dates, act_step1, label="Actual Price", color="black", linewidth=2)
    axes[0].plot(test_dates, pred_step1, label=f"Universal LSTM (RMSE: ${res['rmse']:.2f})", color="#1f77b4", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"{ticker}: Model Prediction vs Ground Truth", fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # Subplot 2: Real-Time Multi-Step Projected Trajectory Cone
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
    Renders an interactive dropdown in Jupyter / Google Colab to switch between companies instantly.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        tickers = sorted(list(eval_results.keys()))
        if not tickers:
            print("[!] No companies available in evaluation results.")
            return

        dropdown = widgets.Dropdown(
            options=tickers,
            value=tickers[0],
            description="Select Stock:",
            layout=widgets.Layout(width="380px")
        )
        out = widgets.Output()

        def on_value_change(change):
            if change["name"] == "value":
                with out:
                    clear_output(wait=True)
                    view_company_forecast(change["new"], eval_results, show_plot=True)

        dropdown.observe(on_value_change, names="value")
        display(dropdown)
        with out:
            view_company_forecast(tickers[0], eval_results, show_plot=True)
        display(out)
    except ImportError:
        print("[!] ipywidgets or IPython not installed. Use view_company_forecast('TICKER', eval_results) directly.")


# =====================================================================
# 5. CLI RUNNER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Universal Multi-Company Stock LSTM Pipeline")
    parser.add_argument("--data_dir", type=str, default="./dataset/archive/NifSent/NIFTY 50", help="Folder containing company CSV files")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for master pooled data")
    parser.add_argument("--forecast_horizon", type=int, default=5, help="Number of candles ahead to forecast")
    parser.add_argument("--seq_length", type=int, default=24, help="Sequence lookback window")
    parser.add_argument("--max_companies", type=int, default=None, help="Optional limit on number of companies to ingest")
    parser.add_argument("--select_ticker", type=str, default="RELIANCE", help="Company ticker to display after training")
    parser.add_argument("--output_dir", type=str, default="./outputs/universal_multi_company", help="Output directory")
    parser.add_argument("--view_only", action="store_true", help="Skip training and view forecast from cached evaluation")
    args = parser.parse_args()

    import pickle
    if args.view_only:
        cache_path = os.path.join(args.output_dir, "eval_results.pkl")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found at '{cache_path}'. Please run training first without --view_only.")
        with open(cache_path, "rb") as f:
            eval_results = pickle.load(f)
        chart_file = os.path.join(args.output_dir, f"{args.select_ticker.upper()}_forecast.png")
        view_company_forecast(args.select_ticker, eval_results, save_path=chart_file, show_plot=False)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing Universal Multi-Company LSTM on device: {device.upper()}")

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_20", "EMA_12", "RSI_14", "MACD", "BB_Width", "Volatility_20",
    ]

    # 1. Ingest all company CSVs and pool into unified dataset
    pooled_data = prepare_multi_company_dataset(
        folder_path=args.data_dir,
        feature_cols=features,
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
        max_companies=args.max_companies,
    )

    # 2. Train ONE universal model on all companies
    summary = train_universal_model(
        pooled_data=pooled_data,
        hidden_size=96,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
    )

    # 3. View the selected company's stock forecast
    eval_results = summary["company_evaluations"]
    target_ticker = args.select_ticker.upper()
    if target_ticker not in eval_results and len(eval_results) > 0:
        target_ticker = list(eval_results.keys())[0]

    chart_file = os.path.join(args.output_dir, f"{target_ticker}_forecast.png")
    view_company_forecast(target_ticker, eval_results, save_path=chart_file, show_plot=False)


if __name__ == "__main__":
    main()
