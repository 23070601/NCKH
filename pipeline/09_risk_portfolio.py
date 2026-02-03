"""
Risk estimation (VaR, CVaR) and CVaR-based portfolio optimization.
"""
import os
import numpy as np
import pandas as pd

from src.utils.risk_metrics import summarize_risk_by_symbol

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False


def load_returns(features_path: str) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    df = df.sort_values(["Symbol", "Date"])
    df["Return"] = df.groupby("Symbol")["Close"].apply(lambda x: np.log(x / x.shift(1)))
    return df


def compute_risk_metrics(df: pd.DataFrame, output_dir: str):
    risk_df = summarize_risk_by_symbol(df, symbol_col="Symbol", return_col="Return")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "risk_metrics.csv")
    risk_df.to_csv(out_path, index=False)
    print(f"✓ Saved risk metrics: {out_path}")


def optimize_portfolio_cvar(df: pd.DataFrame, output_dir: str, alpha: float = 0.95, window: int = 252):
    if not HAS_CVXPY:
        print("⚠ CVXPY not installed. Skipping portfolio optimization.")
        return

    # Build return matrix (T x N)
    df = df.dropna(subset=["Return"]).copy()
    symbols = sorted(df["Symbol"].unique())
    pivot = df.pivot(index="Date", columns="Symbol", values="Return").dropna()

    if len(pivot) < 60:
        print("⚠ Not enough data for portfolio optimization.")
        return

    R = pivot.tail(window).values
    mu = np.nanmean(R, axis=0)
    T, N = R.shape

    # CVaR optimization
    w = cp.Variable(N)
    t = cp.Variable()
    u = cp.Variable(T)

    portfolio_returns = R @ w
    cvar = t + (1 / ((1 - alpha) * T)) * cp.sum(u)

    target_return = np.nanmedian(mu)

    constraints = [
        u >= 0,
        u >= -portfolio_returns - t,
        cp.sum(w) == 1,
        w >= 0,
        mu @ w >= target_return,
    ]

    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if w.value is None:
        print("⚠ Optimization failed.")
        return

    weights = pd.DataFrame({"Symbol": symbols, "Weight": w.value})
    weights = weights.sort_values("Weight", ascending=False)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "portfolio_cvar.csv")
    weights.to_csv(out_path, index=False)
    print(f"✓ Saved portfolio weights: {out_path}")

    summary = {
        "alpha": alpha,
        "target_return": float(target_return),
        "objective_cvar": float(cvar.value),
        "num_assets": int(N),
        "num_days": int(T),
    }
    with open(os.path.join(output_dir, "portfolio_cvar_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    features_path = "data/features/all_features_raw.csv"
    output_dir = "data/results/portfolio"

    df = load_returns(features_path)
    compute_risk_metrics(df, output_dir)
    optimize_portfolio_cvar(df, output_dir)
