import argparse
from pathlib import Path
import sys
import glob
import os

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from datasets.VNStocksDataset import get_graph_in_pyg_format  # noqa: E402


def build_tensors(
    values_path: Path,
    adj_path: Path,
    output_dir: Path,
    past_window: int,
    future_window: int,
    volatility_window: int,
    force: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for f in glob.glob(str(output_dir / "timestep_*.pt")):
            try:
                os.remove(f)
            except OSError:
                pass

    x, close_prices, edge_index, edge_weight = get_graph_in_pyg_format(
        values_path=str(values_path),
        adj_path=str(adj_path),
    )

    returns = torch.diff(torch.log(close_prices), dim=1)
    volatility = torch.zeros_like(close_prices)
    for i in range(volatility_window, close_prices.shape[1]):
        volatility[:, i] = returns[:, i - volatility_window : i].std(dim=1)

    start_idx = max(volatility_window, 0)
    count = 0
    for idx in range(start_idx, x.shape[2] - past_window - future_window):
        future_vol_start = idx + past_window
        future_vol_end = min(future_vol_start + future_window, volatility.shape[1])
        future_volatility = volatility[:, future_vol_start:future_vol_end].mean(dim=1, keepdim=True)

        data = Data(
            x=x[:, :, idx : idx + past_window],
            edge_index=edge_index,
            edge_weight=edge_weight,
            close_price=close_prices[:, idx : idx + past_window],
            y=future_volatility,
            close_price_y=close_prices[:, idx + past_window : idx + past_window + future_window],
            volatility=volatility[:, idx : idx + past_window],
        )

        torch.save(data, output_dir / f"timestep_{count}.pt")
        count += 1

    print(f"✓ Saved {count} timesteps to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build timestep tensors (.pt) for model training")
    parser.add_argument("--values", default="data/raw/values.csv", help="Path to values.csv")
    parser.add_argument("--adj", default="data/raw/adj.npy", help="Path to adj.npy")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--past-window", type=int, default=25, help="Past window length")
    parser.add_argument("--future-window", type=int, default=5, help="Future window length")
    parser.add_argument("--vol-window", type=int, default=20, help="Volatility window")
    parser.add_argument("--force", action="store_true", help="Remove old timesteps before saving")
    args = parser.parse_args()

    build_tensors(
        values_path=Path(args.values),
        adj_path=Path(args.adj),
        output_dir=Path(args.output_dir),
        past_window=args.past_window,
        future_window=args.future_window,
        volatility_window=args.vol_window,
        force=args.force,
    )


if __name__ == "__main__":
    main()
