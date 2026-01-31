"""
Model tuning utilities for volatility regression and risk classification.
Uses latest predictions dataset as a feature table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier


@dataclass
class TuningConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    n_iter: int = 12


def load_latest_predictions(analysis_dir: str | Path) -> pd.DataFrame:
    analysis_dir = Path(analysis_dir)
    files = sorted(analysis_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.csv found in {analysis_dir}")
    return pd.read_csv(files[-1])


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = {
        "Date",
        "Symbol",
        "Future_Vol_5D",
        "Risk_Label",
        "Predicted_Vol_RF",
        "Predicted_Vol_XGB",
        "Predicted_Risk",
        "Unnamed: 0",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    y_reg = df["Future_Vol_5D"].values
    y_cls = df["Risk_Label"].values if "Risk_Label" in df.columns else None

    return X, y_reg, y_cls


def _time_split(df: pd.DataFrame, date_col: str, test_size: float, val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]


def tune_regression(df: pd.DataFrame, cfg: TuningConfig) -> Dict:
    train_df, val_df, test_df = _time_split(df, "Date", cfg.test_size, cfg.val_size)
    X_train, y_train, _ = _prepare_features(train_df)
    X_val, y_val, _ = _prepare_features(val_df)
    X_test, y_test, _ = _prepare_features(test_df)

    # Random Forest
    rf = RandomForestRegressor(random_state=cfg.random_state, n_jobs=-1)
    rf_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [8, 12, 16, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=cfg.n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)

    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)

    # HistGradientBoosting
    hgb = HistGradientBoostingRegressor(random_state=cfg.random_state)
    hgb_params = {
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [200, 400, 600],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    hgb_search = RandomizedSearchCV(
        hgb,
        hgb_params,
        n_iter=cfg.n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=0,
    )
    hgb_search.fit(X_train, y_train)

    hgb_best = hgb_search.best_estimator_
    hgb_pred = hgb_best.predict(X_test)

    results = {
        "rf": {
            "best_params": rf_search.best_params_,
            "rmse": float(mean_squared_error(y_test, rf_pred, squared=False)),
            "mae": float(mean_absolute_error(y_test, rf_pred)),
            "r2": float(r2_score(y_test, rf_pred)),
        },
        "hgb": {
            "best_params": hgb_search.best_params_,
            "rmse": float(mean_squared_error(y_test, hgb_pred, squared=False)),
            "mae": float(mean_absolute_error(y_test, hgb_pred)),
            "r2": float(r2_score(y_test, hgb_pred)),
        },
    }

    return results, rf_best, hgb_best


def tune_classification(df: pd.DataFrame, cfg: TuningConfig) -> Dict:
    if "Risk_Label" not in df.columns:
        return {"error": "Risk_Label not found"}, None, None

    train_df, val_df, test_df = _time_split(df, "Date", cfg.test_size, cfg.val_size)
    X_train, _, y_train = _prepare_features(train_df)
    X_test, _, y_test = _prepare_features(test_df)

    # Random Forest (balanced)
    rf = RandomForestClassifier(random_state=cfg.random_state, n_jobs=-1, class_weight="balanced")
    rf_params = {
        "n_estimators": [200, 400, 600],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=cfg.n_iter,
        scoring="f1_macro",
        cv=3,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)

    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)

    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(random_state=cfg.random_state)
    hgb_params = {
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_iter": [200, 400, 600],
        "l2_regularization": [0.0, 0.1, 1.0],
    }
    hgb_search = RandomizedSearchCV(
        hgb,
        hgb_params,
        n_iter=cfg.n_iter,
        scoring="f1_macro",
        cv=3,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbose=0,
    )
    hgb_search.fit(X_train, y_train)

    hgb_best = hgb_search.best_estimator_
    hgb_pred = hgb_best.predict(X_test)

    results = {
        "rf": {
            "best_params": rf_search.best_params_,
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "f1_macro": float(f1_score(y_test, rf_pred, average="macro")),
        },
        "hgb": {
            "best_params": hgb_search.best_params_,
            "accuracy": float(accuracy_score(y_test, hgb_pred)),
            "f1_macro": float(f1_score(y_test, hgb_pred, average="macro")),
        },
    }

    return results, rf_best, hgb_best
