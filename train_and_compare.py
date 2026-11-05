"""
Model Training Engine, Feature Comparison Experiments, and Visualization Suite.
"""

import os
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lstm_model import StockLSTM
from data_pipeline import StockDataset, prepare_datasets


def train_lstm_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int = 10,
) -> Dict[str, List[float]]:
    """
    Train an LSTM model with MSE loss, Adam optimizer, and early stopping.
    """
    model.to(device)
    criterion = nn.MSELoss()
    # AdamW decouples weight decay from gradient updates for true L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_train_loss += loss.item() * len(X_batch)

        train_loss = running_train_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                running_val_loss += loss.item() * len(X_batch)

        val_loss = running_val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Early stop reached
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history


def evaluate_predictions(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: Any,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Run model on test set, inverse transform to real currency, and compute multi-step regression metrics.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        scaled_preds = model(X_tensor).cpu().numpy()

    forecast_horizon = y_test.shape[1] if y_test.ndim > 1 else 1

    # Inverse transform to original price scale (reshaped to handle arbitrary N-step horizon)
    actuals = target_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, forecast_horizon)
    predictions = target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).reshape(-1, forecast_horizon)

    # Global metrics across all horizon steps
    mae = np.mean(np.abs(actuals - predictions))
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-9))) * 100

    # Step-by-step metrics per horizon step (t+1, t+2, ..., t+N)
    step_rmse = [
        float(np.sqrt(np.mean((actuals[:, s] - predictions[:, s]) ** 2)))
        for s in range(forecast_horizon)
    ]
    step_mae = [
        float(np.mean(np.abs(actuals[:, s] - predictions[:, s])))
        for s in range(forecast_horizon)
    ]

    # Directional Accuracy (% of times predicting correct trajectory direction)
    if forecast_horizon == 1:
        actual_dir = np.diff(actuals.flatten()) > 0
        pred_dir = np.diff(predictions.flatten()) > 0
    else:
        # Net direction from step 0 to step N-1
        actual_dir = (actuals[:, -1] - actuals[:, 0]) > 0
        pred_dir = (predictions[:, -1] - predictions[:, 0]) > 0
    dir_acc = np.mean(actual_dir == pred_dir) * 100 if len(actual_dir) > 0 else 0.0

    return {
        "actuals": actuals,
        "predictions": predictions,
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "step_rmse": step_rmse,
        "step_mae": step_mae,
        "directional_accuracy": float(dir_acc),
        "forecast_horizon": forecast_horizon,
    }


def format_price(v: float) -> str:
    """Format price labels cleanly across stocks, crypto (BTC $80k), and forex (EUR/USD 1.08)."""
    if abs(v) < 0.01:
        return f"${v:.5f}"
    elif abs(v) < 5.0:
        return f"${v:.4f}"
    elif abs(v) < 1000.0:
        return f"${v:.2f}"
    else:
        return f"${v:,.1f}"


def plot_feature_correlations(df: pd.DataFrame, feature_cols: List[str], save_path: str = "feature_correlations.png"):
    """
    Plot correlation matrix heatmap of selected features with the Close price.
    """
    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr().fillna(0.0)
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Pearson Correlation")
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
    plt.yticks(range(len(feature_cols)), feature_cols)
    plt.title("Stock Feature Correlation Matrix", fontsize=14, fontweight="bold")

    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr.iloc[i, j]
            val_num = 0.0 if np.isnan(val) else float(val)
            color = "white" if abs(val_num) > 0.5 else "black"
            plt.text(j, i, f"{val_num:.2f}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved correlation graph to {save_path}")


def plot_market_features(df: pd.DataFrame, ticker: str, save_path: str = "market_features.png"):
    """
    Multi-panel plot visualizing raw price, moving averages, volume, RSI, and MACD.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    recent = df.iloc[-150:]  # Focus on recent 150 trading days for clarity

    # 1. Price + Moving Averages + Bollinger Bands
    ax0 = axes[0]
    ax0.plot(recent.index, recent["Close"], label="Close Price", color="#1f77b4", linewidth=1.8)
    if "SMA_20" in recent.columns:
        ax0.plot(recent.index, recent["SMA_20"], label="SMA 20", color="#ff7f0e", linestyle="--", alpha=0.8)
    if "BB_Upper" in recent.columns and "BB_Lower" in recent.columns:
        ax0.plot(recent.index, recent["BB_Upper"], label="BB Upper", color="gray", linestyle=":", alpha=0.6)
        ax0.plot(recent.index, recent["BB_Lower"], label="BB Lower", color="gray", linestyle=":", alpha=0.6)
        ax0.fill_between(recent.index, recent["BB_Lower"], recent["BB_Upper"], color="gray", alpha=0.1)
    ax0.set_title(f"{ticker} Live Feature Breakdown (Price, Trend, Indicators)", fontsize=13, fontweight="bold")
    ax0.set_ylabel("Price ($)")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.3)

    # 2. Volume
    ax1 = axes[1]
    ax1.bar(recent.index, recent["Volume"], color="#2ca02c", alpha=0.7, width=0.8)
    ax1.set_ylabel("Volume")
    ax1.grid(True, alpha=0.3)

    # 3. RSI
    ax2 = axes[2]
    if "RSI_14" in recent.columns:
        ax2.plot(recent.index, recent["RSI_14"], color="#9467bd", label="RSI 14")
        ax2.axhline(70, color="red", linestyle="--", alpha=0.6, label="Overbought (70)")
        ax2.axhline(30, color="green", linestyle="--", alpha=0.6, label="Oversold (30)")
        ax2.set_ylabel("RSI")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    # 4. MACD
    ax3 = axes[3]
    if "MACD" in recent.columns and "MACD_Signal" in recent.columns:
        ax3.plot(recent.index, recent["MACD"], color="#d62728", label="MACD")
        ax3.plot(recent.index, recent["MACD_Signal"], color="#1f77b4", label="Signal")
        ax3.bar(recent.index, recent["MACD_Hist"], color="gray", alpha=0.5, label="Hist")
        ax3.set_ylabel("MACD")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved market features graph to {save_path}")


def run_feature_comparison_experiments(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    target_col: str = "Close",
    seq_length: int = 30,
    forecast_horizon: int = 1,
    epochs: int = 40,
    batch_size: int = 32,
    hidden_size: int = 64,
    device: str = "cpu",
    output_dir: str = "./outputs",
) -> Dict[str, Any]:
    """
    Trains separate LSTM models on each feature set and compares their performance visually.
    Supports multi-step forecasting for N candles ahead.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, features in feature_sets.items():
        print(f"\n==================================================")
        print(f"[*] Training LSTM on Feature Set: '{name}'")
        print(f"    Features ({len(features)}): {features} | Horizon: {forecast_horizon} candles")
        print(f"==================================================")

        prep = prepare_datasets(
            df=df,
            feature_cols=features,
            target_col=target_col,
            seq_length=seq_length,
            forecast_horizon=forecast_horizon,
        )

        train_dataset = StockDataset(prep["X_train"], prep["y_train"])
        val_dataset = StockDataset(prep["X_val"], prep["y_val"])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = StockLSTM(
            input_size=len(features),
            hidden_size=hidden_size,
            num_layers=2,
            output_size=forecast_horizon,
            dropout=0.2,
        )

        history = train_lstm_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=0.001,
            device=device,
        )

        eval_res = evaluate_predictions(
            model=model,
            X_test=prep["X_test"],
            y_test=prep["y_test"],
            target_scaler=prep["target_scaler"],
            device=device,
        )

        # Save model checkpoint
        ckpt_path = os.path.join(output_dir, f"lstm_{name.replace(' ', '_').lower()}.pt")
        torch.save(
            {
                "model_state": model.state_dict(),
                "feature_cols": features,
                "feature_scaler": prep["feature_scaler"],
                "target_scaler": prep["target_scaler"],
                "seq_length": seq_length,
                "forecast_horizon": forecast_horizon,
                "hidden_size": hidden_size,
            },
            ckpt_path,
        )

        results[name] = {
            "model": model,
            "history": history,
            "metrics": eval_res,
            "prep": prep,
            "ckpt_path": ckpt_path,
        }

        print(f"    -> Overall RMSE: {format_price(eval_res['rmse'])} | MAE: {format_price(eval_res['mae'])} | MAPE: {eval_res['mape']:.2f}% | Dir Acc: {eval_res['directional_accuracy']:.1f}%")
        if forecast_horizon > 1:
            print(f"       Step 1 RMSE: {format_price(eval_res['step_rmse'][0])} -> Step {forecast_horizon} RMSE: {format_price(eval_res['step_rmse'][-1])}")

    # Plot Comparison Graphs
    plot_comparison_dashboard(results, output_dir=output_dir)

    return results


def plot_comparison_dashboard(results: Dict[str, Any], output_dir: str):
    """
    Creates comprehensive comparative charts:
    1. Actual vs Predicted price overlay.
    2. Multi-step trajectory projection & horizon error degradation (if horizon > 1).
    3. Training & Validation loss curves for all models.
    4. Error metric bar charts (RMSE, MAE, Directional Accuracy).
    """
    first_key = list(results.keys())[0]
    test_dates = results[first_key]["prep"]["test_dates"]
    forecast_horizon = results[first_key]["metrics"]["forecast_horizon"]
    actuals = results[first_key]["metrics"]["actuals"]
    actuals_step1 = actuals[:, 0]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 1. Predictions Overlay Chart (Step 1)
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, actuals_step1, label="Ground Truth (Actual Price)", color="black", linewidth=2.5, alpha=0.9)

    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        preds_step1 = data["metrics"]["predictions"][:, 0]
        rmse = data["metrics"]["step_rmse"][0] if forecast_horizon > 1 else data["metrics"]["rmse"]
        lbl_suffix = f" (Step 1 RMSE: {format_price(rmse)})" if forecast_horizon > 1 else f" (RMSE: {format_price(rmse)})"
        plt.plot(
            test_dates,
            preds_step1,
            label=f"{name}{lbl_suffix}",
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
        )

    title_horizon = f" (Forecast Horizon N={forecast_horizon} Candles)" if forecast_horizon > 1 else ""
    plt.title(f"Stock Price Prediction: Multi-Feature Model Comparison on Test Data{title_horizon}", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price", fontsize=11)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    pred_path = os.path.join(output_dir, "feature_comparison_predictions.png")
    plt.savefig(pred_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparative predictions graph to {pred_path}")

    # 2. Dedicated Multi-Step Trajectory & Degradation Chart (when N > 1)
    if forecast_horizon > 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 2A: Horizon Error Degradation Curve
        steps_range = list(range(1, forecast_horizon + 1))
        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            axes[0].plot(steps_range, data["metrics"]["step_rmse"], marker="o", label=name, color=color, linewidth=2)
        axes[0].set_title(f"RMSE Degradation Across Forecast Horizon (1 to {forecast_horizon} Steps Ahead)", fontweight="bold")
        axes[0].set_xlabel("Horizon Step Ahead (t + k)")
        axes[0].set_ylabel("RMSE")
        axes[0].set_xticks(steps_range)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend()

        # Subplot 2B: Visualizing Sample Multi-Step Predicted Trajectories
        sample_idx = min(len(actuals) - 1, int(len(actuals) * 0.9))
        sample_actual_traj = actuals[sample_idx]
        axes[1].plot(steps_range, sample_actual_traj, marker="s", label="Actual Trajectory", color="black", linewidth=2.5)

        for i, (name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            sample_pred_traj = data["metrics"]["predictions"][sample_idx]
            axes[1].plot(steps_range, sample_pred_traj, marker="^", linestyle="--", label=f"{name} Forecast", color=color, linewidth=2)

        axes[1].set_title(f"Sample Multi-Step Trajectory Forecast (Window ending at {test_dates[sample_idx]})", fontweight="bold")
        axes[1].set_xlabel("Forecast Step (t + k)")
        axes[1].set_ylabel("Price")
        axes[1].set_xticks(steps_range)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend()

        plt.tight_layout()
        traj_path = os.path.join(output_dir, "multi_step_trajectories.png")
        plt.savefig(traj_path, dpi=300)
        plt.close()
        print(f"[+] Saved multi-step trajectories graph to {traj_path}")

    # 3. Loss Curves Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        epochs_range = range(1, len(data["history"]["train_loss"]) + 1)
        axes[0].plot(epochs_range, data["history"]["train_loss"], label=name, color=color, linewidth=1.5)
        axes[1].plot(epochs_range, data["history"]["val_loss"], label=name, color=color, linewidth=1.5)

    axes[0].set_title("Training Loss (MSE) Across Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    axes[1].set_title("Validation Loss (MSE) Across Epochs", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    loss_path = os.path.join(output_dir, "feature_comparison_loss.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[+] Saved loss comparison graph to {loss_path}")

    # 3. Bar Chart of Metrics
    names = list(results.keys())
    rmse_vals = [results[k]["metrics"]["rmse"] for k in names]
    mae_vals = [results[k]["metrics"]["mae"] for k in names]
    dir_accs = [results[k]["metrics"]["directional_accuracy"] for k in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # RMSE
    axes[0].bar(names, rmse_vals, color="#1f77b4", alpha=0.8)
    axes[0].set_title("Root Mean Squared Error (Lower is Better)", fontweight="bold")
    axes[0].set_ylabel("RMSE")
    max_rmse = max(rmse_vals) if rmse_vals else 1.0
    for i, v in enumerate(rmse_vals):
        axes[0].text(i, v + (max_rmse * 0.02), format_price(v), ha="center", fontweight="bold")
    axes[0].tick_params(axis="x", rotation=25)

    # MAE
    axes[1].bar(names, mae_vals, color="#2ca02c", alpha=0.8)
    axes[1].set_title("Mean Absolute Error (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel("MAE")
    max_mae = max(mae_vals) if mae_vals else 1.0
    for i, v in enumerate(mae_vals):
        axes[1].text(i, v + (max_mae * 0.02), format_price(v), ha="center", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=25)

    # Directional Accuracy
    axes[2].bar(names, dir_accs, color="#ff7f0e", alpha=0.8)
    axes[2].set_title("Directional Movement Accuracy (Higher is Better)", fontweight="bold")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(0, 100)
    for i, v in enumerate(dir_accs):
        axes[2].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, "feature_comparison_metrics.png")
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"[+] Saved metrics comparison bar chart to {metrics_path}")
