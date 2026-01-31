#!/usr/bin/env python3
"""
Run a simple backtest using predicted volatility/risk for VN FDI stocks.
"""

import argparse
from pathlib import Path
import importlib.util
import os
import sys

# Load backtest module directly to avoid heavy package imports
backtest_path = os.path.join(os.path.dirname(__file__), "src", "utils", "backtest.py")
spec = importlib.util.spec_from_file_location("backtest", backtest_path)
backtest = importlib.util.module_from_spec(spec)
sys.modules["backtest"] = backtest
spec.loader.exec_module(backtest)

BacktestConfig = backtest.BacktestConfig
load_latest_predictions = backtest.load_latest_predictions
plot_results = backtest.plot_results
run_backtest = backtest.run_backtest
save_results = backtest.save_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest volatility/risk-based portfolio selection")
    parser.add_argument("--analysis-dir", default="data/analysis", help="Directory containing predictions_*.csv")
    parser.add_argument("--output-dir", default="data/analysis/backtest", help="Output directory for results")
    parser.add_argument("--horizon", type=int, default=5, help="Forward return horizon in days")
    parser.add_argument("--rebalance", type=int, default=5, help="Rebalance frequency in days")
    parser.add_argument("--topk", default="5,10,20", help="Comma-separated top-k list")
    parser.add_argument(
        "--strategy",
        default="low_risk_low_vol",
        choices=["low_risk_low_vol", "low_vol", "low_risk", "risk_score"],
        help="Selection strategy",
    )
    parser.add_argument("--risk-weight", type=float, default=0.6, help="Risk weight for risk_score strategy")
    parser.add_argument("--vol-weight", type=float, default=0.4, help="Volatility weight for risk_score strategy")
    parser.add_argument("--plot", default="data/analysis/backtest/cumulative_returns.png", help="Output plot path")
    args = parser.parse_args()

    topks = tuple(int(x.strip()) for x in args.topk.split(",") if x.strip())

    cfg = BacktestConfig(
        horizon_days=args.horizon,
        rebalance_every=args.rebalance,
        topks=topks,
        strategy=args.strategy,
        risk_weight=args.risk_weight,
        vol_weight=args.vol_weight,
    )

    df = load_latest_predictions(args.analysis_dir)
    results = run_backtest(df, cfg)

    save_results(results, args.output_dir)
    plot_results(results, args.plot)

    print(f"✓ Backtest complete. Results in {Path(args.output_dir).resolve()}")
    print(f"✓ Plot saved to {Path(args.plot).resolve()}")


if __name__ == "__main__":
    main()
