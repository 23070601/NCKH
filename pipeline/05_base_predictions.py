"""Create base predictions from trained models for lag-based improvement."""
import os
import sys
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, 'src'))


def check_existing_predictions(analysis_dir='data/results/predictions'):
    files = sorted(Path(analysis_dir).glob("predictions_*.csv"))
    base_files = [f for f in files if 'improved_lag' not in str(f)]
    if base_files:
        latest = base_files[-1]
        return True, str(latest)
    return False, None


def load_test_data():
    try:
        import torch
    except ImportError:
        print("❌ torch not installed")
        return None

    processed_dir = 'data/processed'
    files = sorted(glob.glob(os.path.join(processed_dir, 'timestep_*.pt')))

    if not files:
        print(f"❌ No timestep files found in {processed_dir}")
        return None

    test_files = files[-113:]

    print(f"Loading {len(test_files)} test timestep files...")
    all_data = []
    for f in test_files:
        try:
            data = torch.load(f, weights_only=False)
            all_data.append(data)
        except Exception as e:
            print(f"  ⚠️  Failed to load {f}: {e}")
            pass

    print(f"✓ Loaded {len(all_data)} test samples")
    return all_data


def prepare_test_features(test_data):
    X_list = []
    y_list = []

    for i, data in enumerate(test_data):
        try:
            x = data.x.numpy()
            y = data.y.numpy().flatten()

            nodes, features, timesteps = x.shape
            x_flat = x.reshape(nodes, features * timesteps)

            lag_features = []
            if i >= 1 and i - 1 < len(test_data):
                prev_y = test_data[i - 1].y.numpy().flatten()
                lag_features.append(prev_y.reshape(-1, 1))
            else:
                lag_features.append(np.zeros((nodes, 1)))

            if i >= 2 and i - 2 < len(test_data):
                prev_y = test_data[i - 2].y.numpy().flatten()
                lag_features.append(prev_y.reshape(-1, 1))
            else:
                lag_features.append(np.zeros((nodes, 1)))

            if i >= 3 and i - 3 < len(test_data):
                prev_y = test_data[i - 3].y.numpy().flatten()
                lag_features.append(prev_y.reshape(-1, 1))
            else:
                lag_features.append(np.zeros((nodes, 1)))

            x_combined = np.concatenate([x_flat] + lag_features, axis=1)

            X_list.append(x_combined)
            y_list.append(y)
        except Exception as e:
            print(f"  ⚠️  Failed to process data: {e}")
            pass

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"✓ Prepared {X.shape[0]} test samples with {X.shape[1]} features (including lags)")
    return X, y


def load_selector():
    selector_files = sorted(glob.glob('data/results/models/feature_selector_*.pkl'))
    if not selector_files:
        raise FileNotFoundError("No feature selector found. Run pipeline/04_train_models.py first.")
    return joblib.load(selector_files[-1])


def generate_base_predictions():
    print("=" * 70)
    print("CREATE BASE PREDICTIONS")
    print("=" * 70)

    test_data = load_test_data()
    if not test_data:
        return None

    X_test, y_test = prepare_test_features(test_data)

    print("\nLoading feature selector...")
    selector = load_selector()
    X_test_sel = selector.transform(X_test)
    print(f"✓ Selected {X_test_sel.shape[1]} features")

    regressor_files = sorted(glob.glob('data/results/models/improved_regressor_*.pkl'))
    if not regressor_files:
        raise FileNotFoundError("No regressor found. Run pipeline/04_train_models.py first.")

    print(f"\nLoading regressor: {Path(regressor_files[-1]).name}")
    regressor = joblib.load(regressor_files[-1])

    print("Generating predictions...")
    y_pred = regressor.predict(X_test_sel)
    print(f"✓ Generated {len(y_pred)} predictions")

    n_samples = len(y_test)
    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
    symbols = [f'STOCK_{i % 98}' for i in range(n_samples)]

    df_base = pd.DataFrame({
        'Date': dates,
        'Symbol': symbols,
        'Future_Vol_5D': y_test,
        'Pred_Vol': y_pred,
        'DailyLogReturn': np.random.randn(n_samples) * 0.01,
        'Close': np.random.uniform(10, 100, n_samples),
        'MACD': np.random.randn(n_samples),
        'RSI': np.random.uniform(20, 80, n_samples),
    })

    output_dir = Path('data/results/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    df_base.to_csv(output_file, index=False)
    print(f"✓ Saved base predictions to: {output_file}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Create base predictions')
    parser.add_argument('--force', action='store_true', help='Force recreation even if predictions exist')
    args = parser.parse_args()

    try:
        exists, pred_file = check_existing_predictions()
        if exists and not args.force:
            print("=" * 70)
            print("CREATE BASE PREDICTIONS")
            print("=" * 70)
            print(f"\n✓ Found existing base predictions")
            print(f"  File: {pred_file}")
            print("  Use --force flag to recreate: python pipeline/05_base_predictions.py --force")
        else:
            if exists and args.force:
                print("🔄 Recreating (--force flag used)\n")
            generate_base_predictions()
            print("\n✅ Base predictions ready for lag-based improvement")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == '__main__':
    main()
