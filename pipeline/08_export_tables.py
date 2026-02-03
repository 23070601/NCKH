import argparse
from pathlib import Path
import glob
import os
import pandas as pd
import torch


def export_labels(processed_dir: Path, output_dir: Path):
    files = sorted(processed_dir.glob("timestep_*.pt"))
    rows = []
    for idx, f in enumerate(files):
        data = torch.load(f, weights_only=False)
        y = data.y.detach().cpu().numpy().flatten()
        for stock_id, vol in enumerate(y):
            rows.append({
                "Timestep": idx,
                "Stock_ID": f"STOCK_{stock_id}",
                "Volatility": float(vol),
            })
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "all_volatility_labels.csv"
    df.to_csv(out_file, index=False)
    print(f"✓ Saved labels: {out_file}")


def export_timesteps(processed_dir: Path, output_dir: Path):
    files = sorted(processed_dir.glob("timestep_*.pt"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, f in enumerate(files):
        data = torch.load(f, weights_only=False)
        x = data.x.detach().cpu().numpy()  # (stocks, features, days)
        y = data.y.detach().cpu().numpy().flatten()

        stocks, features, days = x.shape
        flat_rows = []
        for stock_id in range(stocks):
            row = {"Stock_ID": f"STOCK_{stock_id}"}
            for feat in range(features):
                for day in range(days):
                    row[f"F{feat}_D{day}"] = float(x[stock_id, feat, day])
            row["Volatility"] = float(y[stock_id])
            flat_rows.append(row)

        df = pd.DataFrame(flat_rows)
        out_file = output_dir / f"timestep_{idx}.csv"
        df.to_csv(out_file, index=False)

    print(f"✓ Exported {len(files)} timestep tables to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export input/output tables to CSV")
    parser.add_argument("--processed", default="data/processed", help="Processed timesteps folder")
    parser.add_argument("--exports", default="data/results/exports", help="Exports output folder")
    parser.add_argument("--all", action="store_true", help="Export all timestep CSVs")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    exports_dir = Path(args.exports)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed folder not found: {processed_dir}")

    export_labels(processed_dir, exports_dir)

    if args.all:
        export_timesteps(processed_dir, exports_dir / "timesteps")


if __name__ == "__main__":
    main()
