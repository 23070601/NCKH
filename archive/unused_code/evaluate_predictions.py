"""
Evaluate prediction quality for volatility regression and risk classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # sklearn < 1.4
    root_mean_squared_error = None


def load_latest_predictions(analysis_dir: str | Path) -> pd.DataFrame:
    analysis_dir = Path(analysis_dir)
    files = sorted(analysis_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.csv found in {analysis_dir}")
    return pd.read_csv(files[-1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if root_mean_squared_error is not None:
        rmse = root_mean_squared_error(y_true, y_pred)
    else:
        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def evaluate(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    results = {}

    # Volatility regression
    y_true = df["Future_Vol_5D"].values
    results["rf_regression"] = regression_metrics(y_true, df["Predicted_Vol_RF"].values)
    if "Predicted_Vol_XGB" in df.columns:
        results["xgb_regression"] = regression_metrics(y_true, df["Predicted_Vol_XGB"].values)

    # Risk classification
    if "Risk_Label" in df.columns and "Predicted_Risk" in df.columns:
        y_true_cls = df["Risk_Label"].values
        y_pred_cls = df["Predicted_Risk"].values
        cm = confusion_matrix(y_true_cls, y_pred_cls)
        results["risk_classification"] = {
            "accuracy": float((y_true_cls == y_pred_cls).mean()),
            "confusion_matrix": cm.tolist(),
        }

    return results


def plot_confusion_matrix(cm: np.ndarray, output_path: str | Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Risk Classification Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_calibration(df: pd.DataFrame, output_path: str | Path) -> None:
    # Simple reliability curve using risk classes as ordinal proxy
    y_true = df["Risk_Label"].values
    y_pred = df["Predicted_Risk"].values

    bins = [0, 1, 2, 3]
    bin_centers = [0.5, 1.5, 2.5]

    acc = []
    for i in range(3):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            acc.append(np.nan)
        else:
            acc.append((y_true[mask] == y_pred[mask]).mean())

    plt.figure(figsize=(5, 4))
    plt.plot(bin_centers, acc, marker="o", label="Empirical Accuracy")
    plt.plot([0.5, 2.5], [1.0, 1.0], linestyle="--", label="Perfect")
    plt.ylim(0, 1)
    plt.title("Risk Class Calibration (Empirical)")
    plt.xlabel("Predicted Risk Class")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def classification_report_text(df: pd.DataFrame) -> str:
    y_true = df["Risk_Label"].values
    y_pred = df["Predicted_Risk"].values
    return classification_report(y_true, y_pred, digits=4)
