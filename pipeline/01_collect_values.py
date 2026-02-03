import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from VNStocks import VNStocksDataset  # noqa: E402


def collect_values(
    stock_list_path: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    source: str,
):
    dataset = VNStocksDataset(
        stock_list_path=stock_list_path,
        start_date=start_date,
        end_date=end_date,
        raw_dir=str(ROOT / "data" / "raw"),
        processed_dir=output_dir,
    )
    dataset.collect_price_data(source=source)
    dataset.process_and_save()


def main():
    parser = argparse.ArgumentParser(description="Collect raw stock data and build values.csv + adj.npy")
    parser.add_argument("--stock-list", default="data/raw/fdi_stocks_list.csv", help="CSV file of tickers")
    parser.add_argument("--start", default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory for values.csv and adj.npy")
    parser.add_argument("--source", default="manual", help="Data source: manual | vnstock | vndirect")
    args = parser.parse_args()

    collect_values(
        stock_list_path=args.stock_list,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        source=args.source,
    )


if __name__ == "__main__":
    main()
