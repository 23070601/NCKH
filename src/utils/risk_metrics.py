"""
Risk metrics for volatility, VaR, and CVaR.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats


def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95, method: str = "historical") -> float:
    """Compute Value at Risk (VaR). Returns a positive loss value."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return float("nan")

    if method == "historical":
        var = -np.percentile(returns, (1 - confidence_level) * 100)
    elif method == "parametric":
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        var = -(mu + sigma * stats.norm.ppf(1 - confidence_level))
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(var)


def conditional_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """Compute Conditional Value at Risk (CVaR / Expected Shortfall)."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if returns.size == 0:
        return float("nan")

    var = value_at_risk(returns, confidence_level, method="historical")
    tail = returns[returns <= -var]
    if tail.size == 0:
        return float(var)
    return float(-tail.mean())


def summarize_risk_by_symbol(df: pd.DataFrame, symbol_col: str = "Symbol", return_col: str = "Return") -> pd.DataFrame:
    """Compute VaR/CVaR per symbol."""
    rows = []
    for symbol, group in df.groupby(symbol_col):
        ret = group[return_col].dropna().values
        rows.append(
            {
                "Symbol": symbol,
                "Mean_Return": float(np.mean(ret)) if ret.size > 0 else float("nan"),
                "Volatility": float(np.std(ret, ddof=1)) if ret.size > 1 else float("nan"),
                "VaR_95": value_at_risk(ret, 0.95),
                "CVaR_95": conditional_var(ret, 0.95),
            }
        )
    return pd.DataFrame(rows)
