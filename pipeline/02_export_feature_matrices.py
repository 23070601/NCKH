import argparse
from pathlib import Path
import pandas as pd


def export_feature_matrix(values_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    values = pd.read_csv(values_path)
    values = values.set_index(["Symbol", "Date"])

    # Export tickers list
    tickers = values.index.get_level_values("Symbol").unique().tolist()
    pd.DataFrame({"ticker": tickers}).to_csv(output_dir / "tickers.csv", index=False)

    def export_feature(feature_name: str, file_name: str):
        matrix = values[feature_name].unstack(level="Symbol")
        matrix.index.name = "Date"
        matrix.to_csv(output_dir / file_name)

    base_features = [
        ("Open", "open_matrix.csv"),
        ("High", "high_matrix.csv"),
        ("Low", "low_matrix.csv"),
        ("Close", "close_matrix.csv"),
        ("Volume", "volume_matrix.csv"),
        ("DailyLogReturn", "dailylogreturn_matrix.csv"),
        ("RSI", "rsi_matrix.csv"),
        ("MACD", "macd_matrix.csv"),
        ("MA_5", "ma_5_matrix.csv"),
        ("MA_10", "ma_10_matrix.csv"),
        ("MA_20", "ma_20_matrix.csv"),
        ("BB_MID", "bb_mid_matrix.csv"),
        ("BB_UPPER", "bb_upper_matrix.csv"),
        ("BB_LOWER", "bb_lower_matrix.csv"),
        ("VOL_20", "vol_20_matrix.csv"),
    ]

    for feat, fname in base_features:
        if feat in values.columns:
            export_feature(feat, fname)

    # VNIndex (single series)
    if "VNIndex_Close" in values.columns:
        vnindex = values["VNIndex_Close"].unstack(level="Symbol")
        vnindex = vnindex.iloc[:, [0]]
        vnindex.columns = ["VNIndex_Close"]
        vnindex.index.name = "Date"
        vnindex.to_csv(output_dir / "vnindex_close.csv")
    if "VNIndex_Return" in values.columns:
        vnindex_r = values["VNIndex_Return"].unstack(level="Symbol")
        vnindex_r = vnindex_r.iloc[:, [0]]
        vnindex_r.columns = ["VNIndex_Return"]
        vnindex_r.index.name = "Date"
        vnindex_r.to_csv(output_dir / "vnindex_return.csv")

    print("✓ Exported feature matrices to:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export feature matrices from values.csv")
    parser.add_argument("--values", default="data/raw/values.csv", help="Path to values.csv")
    parser.add_argument("--output-dir", default="data/features", help="Output directory")
    args = parser.parse_args()

    export_feature_matrix(Path(args.values), Path(args.output_dir))


if __name__ == "__main__":
    main()
