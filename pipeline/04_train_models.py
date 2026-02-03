import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import glob
import json
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, 'src'))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

from src.models.lstm import LSTMModel

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False


def check_existing_models(output_dir='data/results/models'):
    model_files = sorted(glob.glob(os.path.join(output_dir, 'improved_regressor_*.pkl')))
    if model_files:
        latest = model_files[-1]
        timestamp = os.path.basename(latest).split('_')[2].split('.')[0]
        return True, latest, timestamp
    return False, None, None


def load_data():
    processed_dir = 'data/processed'
    files = sorted(glob.glob(os.path.join(processed_dir, 'timestep_*.pt')))

    print(f"Loading {len(files)} timestep files...")

    all_data = []
    for f in files:
        try:
            data = torch.load(f, weights_only=False)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {str(e)}")

    if len(all_data) == 0:
        raise ValueError(f"No data loaded from {processed_dir}")

    n = len(all_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    print(f"✓ Loaded {len(all_data)} samples (train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)})")

    return train_data, val_data, test_data


def prepare_features_with_lags(data_list, n_lags=3):
    X_list = []
    y_list = []

    for i, data in enumerate(data_list):
        x = data.x.numpy()
        y = data.y.numpy().flatten()

        nodes, features, timesteps = x.shape
        x_flat = x.reshape(nodes, features * timesteps)

        lag_features = []
        for lag in range(1, n_lags + 1):
            if i >= lag:
                prev_data = data_list[i - lag]
                prev_y = prev_data.y.numpy().flatten()
                lag_features.append(prev_y.reshape(-1, 1))
            else:
                lag_features.append(np.zeros((nodes, 1)))

        x_combined = np.concatenate([x_flat] + lag_features, axis=1)

        X_list.append(x_combined)
        y_list.append(y)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    return X, y


def select_top_features(X_train, y_train, X_val, X_test, n_features=15):
    print(f"\nFeature selection: {X_train.shape[1]} -> {n_features} features...")

    selector = SelectKBest(f_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    print(f"✓ Selected {X_train_selected.shape[1]} features")

    return X_train_selected, X_val_selected, X_test_selected, selector


def prepare_sequence_dataset(data_list):
    """Build sequence dataset for LSTM: (samples, timesteps, features)."""
    sequences = []
    targets = []

    for data in data_list:
        x = data.x.numpy()  # (nodes, features, timesteps)
        y = data.y.numpy().flatten()
        nodes, features, timesteps = x.shape

        for i in range(nodes):
            seq = x[i].T  # (timesteps, features)
            sequences.append(seq)
            targets.append(y[i])

    X = np.stack(sequences, axis=0)
    y = np.array(targets)
    return X, y


def normalize_sequences(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def train_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=8, batch_size=256):
    """Train LSTM on sequence data and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[-1]

    model = LSTMModel(in_features=n_features, hidden_size=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze(-1)
                val_losses.append(criterion(preds, yb).item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  LSTM Epoch {epoch}/{epochs} - val_loss: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            pred = model(xb).squeeze(-1).cpu().numpy()
            preds.append(pred)
    y_pred = np.concatenate(preds) if preds else np.array([])

    lstm_r2 = r2_score(y_test, y_pred) if len(y_pred) else float("nan")
    lstm_rmse = np.sqrt(mean_squared_error(y_test, y_pred)) if len(y_pred) else float("nan")

    return model, y_pred, lstm_r2, lstm_rmse


def train_baseline_comparison(X_train, y_train, X_test, y_test):
    print("\nTraining baseline models...")

    results = {}

    mean_pred = np.full(len(y_test), y_train.mean())
    mean_r2 = r2_score(y_test, mean_pred)
    print(f"  Mean baseline: R² = {mean_r2:.4f}")
    results['mean'] = mean_r2

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_r2 = r2_score(y_test, ridge.predict(X_test))
    print(f"  Ridge: R² = {ridge_r2:.4f}")
    results['ridge'] = ridge_r2

    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    print(f"  Random Forest: R² = {rf_r2:.4f}")
    results['rf'] = rf_r2

    return results


def train_improved_models(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nTraining improved models...")

    print("  [1/3] Random Forest (tuned)...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"      RF: R² = {rf_r2:.4f}, RMSE = {rf_rmse:.6f}")

    print("  [2/3] Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
    print(f"      GB: R² = {gb_r2:.4f}, RMSE = {gb_rmse:.6f}")

    print("  [3/4] Ensemble (RF + GB + Ridge)...")
    ridge = Ridge(alpha=1.0)

    ensemble = VotingRegressor([
        ('rf', rf),
        ('gb', gb),
        ('ridge', ridge)
    ])
    ensemble.fit(X_train, y_train)
    ens_pred = ensemble.predict(X_test)
    ens_r2 = r2_score(y_test, ens_pred)
    ens_rmse = np.sqrt(mean_squared_error(y_test, ens_pred))
    print(f"      Ensemble: R² = {ens_r2:.4f}, RMSE = {ens_rmse:.6f}")

    xgb = None
    xgb_r2 = float("nan")
    xgb_rmse = float("nan")
    xgb_pred = None
    if HAS_XGBOOST:
        print("  [4/4] XGBoost...")
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            random_state=42,
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        print(f"      XGB: R² = {xgb_r2:.4f}, RMSE = {xgb_rmse:.6f}")
    else:
        print("  [4/4] XGBoost... (skipped - xgboost not installed)")

    results = {
        'rf': {'model': rf, 'r2': rf_r2, 'rmse': rf_rmse, 'pred': rf_pred},
        'gb': {'model': gb, 'r2': gb_r2, 'rmse': gb_rmse, 'pred': gb_pred},
        'ensemble': {'model': ensemble, 'r2': ens_r2, 'rmse': ens_rmse, 'pred': ens_pred}
    }

    if xgb is not None:
        results['xgb'] = {'model': xgb, 'r2': xgb_r2, 'rmse': xgb_rmse, 'pred': xgb_pred}

    best_model_name = max(results, key=lambda k: results[k]['r2'])
    print(f"  ✓ Best: {best_model_name.upper()} (R² = {results[best_model_name]['r2']:.4f})")

    return results, best_model_name


def train_classification(X_train, y_train_vol, X_test, y_test_vol):
    print("\nTraining risk classifier...")

    p33 = np.percentile(y_train_vol, 33.33)
    p67 = np.percentile(y_train_vol, 66.67)

    y_train_risk = np.array([0 if v <= p33 else (1 if v <= p67 else 2) for v in y_train_vol])
    y_test_risk = np.array([0 if v <= p33 else (1 if v <= p67 else 2) for v in y_test_vol])

    if HAS_SMOTE:
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train_risk)
        except:
            X_train_bal, y_train_bal = X_train, y_train_risk
    else:
        X_train_bal, y_train_bal = X_train, y_train_risk

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_bal, y_train_bal)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test_risk, y_pred)
    f1 = f1_score(y_test_risk, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test_risk, y_pred)

    print(f"  ✓ Accuracy: {acc:.4f} (baseline: 0.333, +{(acc - 0.333)*100:.1f}%)")

    return clf, acc, f1, cm, y_test_risk, y_pred


def save_results(reg_results, best_model_name, clf_model, clf_acc, clf_f1, clf_cm,
                selector, y_test_vol, y_test_risk, y_pred_clf,
                lstm_info=None):
    output_dir = 'data/results/models'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_reg_model = reg_results[best_model_name]['model']
    joblib.dump(best_reg_model, os.path.join(output_dir, f'improved_regressor_{timestamp}.pkl'))
    joblib.dump(clf_model, os.path.join(output_dir, f'improved_classifier_{timestamp}.pkl'))
    joblib.dump(selector, os.path.join(output_dir, f'feature_selector_{timestamp}.pkl'))

    if 'xgb' in reg_results:
        joblib.dump(reg_results['xgb']['model'], os.path.join(output_dir, f'xgb_regressor_{timestamp}.pkl'))

    if lstm_info is not None:
        lstm_model, lstm_mean, lstm_std, lstm_r2, lstm_rmse = lstm_info
        torch.save(lstm_model.state_dict(), os.path.join(output_dir, f'lstm_model_{timestamp}.pt'))
        np.savez(os.path.join(output_dir, f'lstm_scaler_{timestamp}.npz'), mean=lstm_mean, std=lstm_std)

    summary = {
        'timestamp': timestamp,
        'regression': {
            'best_model': best_model_name,
            'rf_r2': float(reg_results['rf']['r2']),
            'rf_rmse': float(reg_results['rf']['rmse']),
            'gb_r2': float(reg_results['gb']['r2']),
            'gb_rmse': float(reg_results['gb']['rmse']),
            'ensemble_r2': float(reg_results['ensemble']['r2']),
            'ensemble_rmse': float(reg_results['ensemble']['rmse']),
            'xgb_r2': float(reg_results['xgb']['r2']) if 'xgb' in reg_results else None,
            'xgb_rmse': float(reg_results['xgb']['rmse']) if 'xgb' in reg_results else None,
            'improvement_vs_baseline': float((reg_results[best_model_name]['r2'] - (-0.015)) * 100),
        },
        'classification': {
            'accuracy': float(clf_acc),
            'f1': float(clf_f1),
            'confusion_matrix': clf_cm.tolist(),
        },
        'lstm': {
            'r2': float(lstm_info[3]) if lstm_info is not None else None,
            'rmse': float(lstm_info[4]) if lstm_info is not None else None,
        }
    }

    with open(os.path.join(output_dir, f'summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save per-sample prediction outputs for analysis
    output_df = pd.DataFrame({
        'y_true_vol': y_test_vol,
        'y_true_risk': y_test_risk,
        'y_pred_risk': y_pred_clf,
    })
    output_df.to_csv(os.path.join(output_dir, f'predictions_{timestamp}.csv'), index=False)

    print(f"\n✓ Saved models and summary to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train models with lag features')
    parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM training')
    args = parser.parse_args()

    exists, latest_model, timestamp = check_existing_models()
    if exists and not args.force:
        print("=" * 70)
        print("TRAINING MODELS")
        print("=" * 70)
        print(f"\n✓ Found existing models")
        print(f"  Latest: {latest_model}")
        print("  Use --force flag to retrain: python pipeline/04_train_models.py --force")
        return

    if exists and args.force:
        print("🔄 Retraining (--force flag used)\n")

    train_data, val_data, test_data = load_data()

    X_train, y_train = prepare_features_with_lags(train_data)
    X_val, y_val = prepare_features_with_lags(val_data)
    X_test, y_test = prepare_features_with_lags(test_data)

    X_train_sel, X_val_sel, X_test_sel, selector = select_top_features(
        X_train, y_train, X_val, X_test, n_features=15
    )

    train_baseline_comparison(X_train_sel, y_train, X_test_sel, y_test)

    reg_results, best_model_name = train_improved_models(
        X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test
    )

    lstm_info = None
    if not args.skip_lstm:
        print("\nTraining LSTM model...")
        X_train_seq, y_train_seq = prepare_sequence_dataset(train_data)
        X_val_seq, y_val_seq = prepare_sequence_dataset(val_data)
        X_test_seq, y_test_seq = prepare_sequence_dataset(test_data)

        X_train_seq, X_val_seq, X_test_seq, mean_seq, std_seq = normalize_sequences(
            X_train_seq, X_val_seq, X_test_seq
        )

        lstm_model, lstm_pred, lstm_r2, lstm_rmse = train_lstm_model(
            X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
            epochs=8, batch_size=256
        )
        print(f"  LSTM: R² = {lstm_r2:.4f}, RMSE = {lstm_rmse:.6f}")
        lstm_info = (lstm_model, mean_seq, std_seq, lstm_r2, lstm_rmse)
    else:
        print("\nSkipping LSTM training (--skip-lstm)")

    clf_model, clf_acc, clf_f1, clf_cm, y_test_risk, y_pred_clf = train_classification(
        X_train_sel, y_train, X_test_sel, y_test
    )

    save_results(reg_results, best_model_name, clf_model, clf_acc, clf_f1, clf_cm,
                selector, y_test, y_test_risk, y_pred_clf, lstm_info=lstm_info)

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
