"""
Backtest utilities for volatility/risk-based portfolio selection.
Uses prediction outputs to simulate simple top-k strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class BacktestConfig:
    horizon_days: int = 5
    rebalance_every: int = 5
    topks: Tuple[int, ...] = (5, 10, 20)
    strategy: str = "low_risk_low_vol"  # options: low_risk_low_vol, low_vol, low_risk, risk_score
    risk_col: str = "Predicted_Risk"
    vol_col: str = "Predicted_Vol_RF"
    date_col: str = "Date"
    symbol_col: str = "Symbol"
    close_col: str = "Close"
    risk_weight: float = 0.6
    vol_weight: float = 0.4


def load_latest_predictions(analysis_dir: str | Path) -> pd.DataFrame:
    analysis_dir = Path(analysis_dir)
    files = sorted(analysis_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions_*.csv found in {analysis_dir}")
    df = pd.read_csv(files[-1])
    return df


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _compute_forward_return(df: pd.DataFrame, date_col: str, symbol_col: str, close_col: str, horizon: int) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df, date_col)
    df.sort_values([symbol_col, date_col], inplace=True)
    df["Close_Fwd"] = df.groupby(symbol_col)[close_col].shift(-horizon)
    df["Forward_Return"] = df["Close_Fwd"] / df[close_col] - 1.0
    return df


def _normalize(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def _select_portfolio(
    df_date: pd.DataFrame,
    k: int,
    strategy: str,
    risk_col: str,
    vol_col: str,
    risk_weight: float,
    vol_weight: float,
) -> pd.DataFrame:
    if strategy == "low_risk_low_vol":
        # Filter low-risk first, then pick lowest volatility. If not enough, fill with next lowest vol.
        low_risk = df_date[df_date[risk_col] == 0]
        if len(low_risk) >= k:
            return low_risk.nsmallest(k, vol_col)
        remainder = df_date[~df_date.index.isin(low_risk.index)]
        combined = pd.concat([low_risk, remainder.nsmallest(k - len(low_risk), vol_col)])
        return combined
    if strategy == "low_risk":
        return df_date.nsmallest(k, risk_col)
    if strategy == "risk_score":
        risk_norm = _normalize(df_date[risk_col])
        vol_norm = _normalize(df_date[vol_col])
        score = risk_weight * risk_norm + vol_weight * vol_norm
        return df_date.loc[score.nsmallest(k).index]
    # default: low volatility
    return df_date.nsmallest(k, vol_col)


def _max_drawdown(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = (arr - peak) / peak
    return float(dd.min())


def _summary_metrics(df: pd.DataFrame, steps_per_year: float) -> Dict[str, float]:
    if df.empty:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
        }
    ret = df["Portfolio_Return"].values
    cum = df["Portfolio_Value"].iloc[-1] - 1.0
    ann_return = (1.0 + ret.mean()) ** steps_per_year - 1.0
    ann_vol = ret.std(ddof=1) * np.sqrt(steps_per_year) if len(ret) > 1 else 0.0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    max_dd = _max_drawdown(df["Portfolio_Value"].values.tolist())
    avg_turnover = df["Turnover"].mean() if "Turnover" in df.columns else 0.0
    return {
        "cumulative_return": float(cum),
        "annualized_return": float(ann_return),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_turnover": float(avg_turnover),
    }


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> Dict[str, pd.DataFrame]:
    df = _compute_forward_return(df, cfg.date_col, cfg.symbol_col, cfg.close_col, cfg.horizon_days)
    df = df.dropna(subset=["Forward_Return"])

    dates = sorted(df[cfg.date_col].unique())
    # rebalance dates every N steps
    rebalance_dates = dates[::cfg.rebalance_every]

    results = {}

    steps_per_year = 252 / max(cfg.rebalance_every, 1)

    for k in cfg.topks:
        portfolio_returns = [1.0]
        market_returns = [1.0]
        records = []
        prev_holdings: Optional[set] = None

        for d in rebalance_dates:
            df_date = df[df[cfg.date_col] == d]
            if df_date.empty:
                continue
            portfolio = _select_portfolio(
                df_date,
                k,
                cfg.strategy,
                cfg.risk_col,
                cfg.vol_col,
                cfg.risk_weight,
                cfg.vol_weight,
            )

            holdings = set(portfolio[cfg.symbol_col].values.tolist())
            if prev_holdings is None:
                turnover = 0.0
            else:
                turnover = 1.0 - (len(holdings.intersection(prev_holdings)) / max(len(holdings), 1))
            prev_holdings = holdings

            # average forward return of portfolio and market
            port_ret = portfolio["Forward_Return"].mean()
            mkt_ret = df_date["Forward_Return"].mean()

            portfolio_returns.append(portfolio_returns[-1] * (1.0 + port_ret))
            market_returns.append(market_returns[-1] * (1.0 + mkt_ret))

            records.append(
                {
                    "Date": d,
                    "Portfolio_Return": port_ret,
                    "Market_Return": mkt_ret,
                    "Portfolio_Value": portfolio_returns[-1],
                    "Market_Value": market_returns[-1],
                    "Turnover": turnover,
                    "TopK": k,
                }
            )

        df_result = pd.DataFrame(records)
        summary = _summary_metrics(df_result, steps_per_year)
        for key, value in summary.items():
            df_result.attrs[key] = value
        results[str(k)] = df_result

    return results


def plot_results(results: Dict[str, pd.DataFrame], output_path: str | Path) -> None:
    plt.figure(figsize=(10, 5))
    for k, df in results.items():
        if df.empty:
            continue
        plt.plot(df["Portfolio_Value"].values, label=f"Top-{k}")
        # plot market once
    if results:
        any_df = next(iter(results.values()))
        if not any_df.empty:
            plt.plot(any_df["Market_Value"].values, label="Market", linewidth=3)
    plt.grid(which="major", linestyle="-", linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.title("Market vs Top-k Portfolio (Volatility/Risk-Based)")
    plt.xlabel("Rebalance Steps")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_results(results: Dict[str, pd.DataFrame], output_dir: str | Path) -> List[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    summary_rows = []
    for k, df in results.items():
        out = output_dir / f"backtest_top{k}.csv"
        df.to_csv(out, index=False)
        paths.append(out)
        summary_rows.append(
            {
                "TopK": int(k),
                "Cumulative_Return": df.attrs.get("cumulative_return", 0.0),
                "Annualized_Return": df.attrs.get("annualized_return", 0.0),
                "Annualized_Volatility": df.attrs.get("annualized_volatility", 0.0),
                "Sharpe": df.attrs.get("sharpe", 0.0),
                "Max_Drawdown": df.attrs.get("max_drawdown", 0.0),
                "Avg_Turnover": df.attrs.get("avg_turnover", 0.0),
            }
        )
    summary_path = output_dir / "backtest_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    paths.append(summary_path)
    return paths
