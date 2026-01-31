import os
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False


def check_existing_improved_predictions(analysis_dir=Path('data/analysis')):
    """Check if improved predictions already exist."""
    files = sorted(analysis_dir.glob("predictions_improved_lag_*.csv"))
    if files:
        latest = files[-1]
        return True, str(latest)
    return False, None


def load_latest_predictions(analysis_dir: Path) -> pd.DataFrame:
    files = sorted(analysis_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No predictions CSV found in {analysis_dir}")
    return pd.read_csv(files[-1])


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Symbol", "Date"], inplace=True)
    for lag in (1, 2, 3):
        df[f"Return_Lag_{lag}"] = df.groupby("Symbol")["DailyLogReturn"].shift(lag)
    df["Return_MA_5"] = df.groupby("Symbol")["DailyLogReturn"].rolling(5).mean().reset_index(level=0, drop=True)
    df["Return_Std_5"] = df.groupby("Symbol")["DailyLogReturn"].rolling(5).std().reset_index(level=0, drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    exclude = {
        "Date",
        "Symbol",
        "Future_Vol_5D",
        "Future_Vol_10D",
        "RiskClass",
        "RiskClass_Pred",
    }
    features = [col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])]
    return df[features], features


def train_improved_regressor(df_base: pd.DataFrame, X_base: pd.DataFrame, y_base: np.ndarray):
    print("\nTraining improved regressor on base predictions...")
    
    df_with_lags = add_lag_features(df_base)
    X_all, all_features = build_feature_matrix(df_with_lags)
    X_all = X_all.fillna(0.0)
    
    selector = SelectKBest(f_regression, k=min(20, X_all.shape[1]))
    X_selected = selector.fit_transform(X_all, y_base)
    
    print(f"  Selected {X_selected.shape[1]} features from {X_all.shape[1]}")
    
    ridge = Ridge(alpha=1.0)
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=18,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    ensemble = VotingRegressor([('ridge', ridge), ('rf', rf), ('gb', gb)])
    ensemble.fit(X_selected, y_base)
    
    return ensemble, selector, df_with_lags


def generate_predictions(analysis_dir: Path):
    print("=" * 70)
    print("GENERATE PREDICTIONS (Lag-Based)")
    print("=" * 70)
    
    print("\nLoading base predictions...")
    df_base = load_latest_predictions(analysis_dir)
    print(f"✓ Loaded {len(df_base)} records")
    
    y_base = df_base['Future_Vol_5D'].values
    
    print("\nBuilding features...")
    df_with_lags = add_lag_features(df_base)
    X_base, _ = build_feature_matrix(df_with_lags)
    X_base = X_base.fillna(0.0)
    
    print(f"✓ {X_base.shape[1]} features, {X_base.shape[0]} samples")
    
    ensemble, selector, df_features = train_improved_regressor(df_base, X_base, y_base)
    
    print("\nGenerating predictions...")
    X_final, final_features = build_feature_matrix(df_features)
    X_final = X_final.fillna(0.0)
    X_selected = selector.transform(X_final)
    
    predictions = ensemble.predict(X_selected)
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    output_csv = analysis_dir / f"predictions_improved_lag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_result = df_base.copy()
    df_result['Pred_Vol'] = predictions
    
    df_result.to_csv(output_csv, index=False)
    print(f"✓ Saved to: {output_csv}")
    
    return str(output_csv), predictions, y_base


def run_backtest_on_predictions(predictions: np.ndarray, y_true: np.ndarray, analysis_dir: Path):
    print("\n" + "=" * 70)
    print("BACKTEST ANALYSIS")
    print("=" * 70)
    
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_dir = analysis_dir / "backtest_improved_lag"
    backtest_dir.mkdir(exist_ok=True)
    
    pred_vol = np.clip(predictions, 0.001, np.inf)
    true_vol = np.clip(y_true, 0.001, np.inf)
    
    r2 = r2_score(true_vol, pred_vol)
    mae = np.mean(np.abs(true_vol - pred_vol))
    rmse = np.sqrt(np.mean((true_vol - pred_vol) ** 2))
    
    print(f"\nRegression Metrics:")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    
    summary = {
        'timestamp': date_str,
        'metrics': {
            'r2': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
        },
        'n_samples': len(predictions),
    }
    
    with open(backtest_dir / f"backtest_summary_{date_str}.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Backtest results saved to: {backtest_dir}")


def cleanup_old_results():
    import glob
    analysis_dir = Path('data/analysis')
    
    # Only cleanup the "improved_lag" results (not base predictions needed as input)
    old_predictions = list(analysis_dir.glob("predictions_improved_lag_*.csv"))
    for f in old_predictions:
        try:
            os.remove(f)
        except:
            pass
    
    old_summaries = list(analysis_dir.glob("backtest_improved_lag/backtest_summary_*.json"))
    for f in old_summaries:
        try:
            os.remove(f)
        except:
            pass
    
    if old_predictions or old_summaries:
        print(f"🧹 Cleaned {len(old_predictions) + len(old_summaries)} old improved results")


def main():
    parser = argparse.ArgumentParser(description='Generate improved predictions')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if predictions exist')
    args = parser.parse_args()
    
    analysis_dir = Path('data/analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    # Check for existing improved predictions
    exists, pred_file = check_existing_improved_predictions(analysis_dir)
    if exists and not args.force:
        print("=" * 70)
        print("GENERATE PREDICTIONS (Lag-Based)")
        print("=" * 70)
        print(f"\n✓ Found existing improved predictions")
        print(f"  File: {pred_file}")
        print(f"  Use --force flag to regenerate: python generate_predictions.py --force")
        return
    
    if exists and args.force:
        print("🔄 Regenerating (--force flag used)\n")
    
    cleanup_old_results()
    
    try:
        pred_file, predictions, y_true = generate_predictions(analysis_dir)
        run_backtest_on_predictions(predictions, y_true, analysis_dir)
        
        print("\n" + "=" * 70)
        print("✅ COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == '__main__':
    main()
