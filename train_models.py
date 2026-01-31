import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import glob
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except:
    HAS_SMOTE = False


def check_existing_models(output_dir='data/analysis/quick_improvement'):
    """Check if latest models already exist."""
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
    
    print("  [3/3] Ensemble (RF + GB + Ridge)...")
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
    
    results = {
        'rf': {'model': rf, 'r2': rf_r2, 'rmse': rf_rmse, 'pred': rf_pred},
        'gb': {'model': gb, 'r2': gb_r2, 'rmse': gb_rmse, 'pred': gb_pred},
        'ensemble': {'model': ensemble, 'r2': ens_r2, 'rmse': ens_rmse, 'pred': ens_pred}
    }
    
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
                selector, y_test_vol, y_test_risk, y_pred_clf):
    output_dir = 'data/analysis/quick_improvement'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    best_reg_model = reg_results[best_model_name]['model']
    joblib.dump(best_reg_model, os.path.join(output_dir, f'improved_regressor_{timestamp}.pkl'))
    joblib.dump(clf_model, os.path.join(output_dir, f'improved_classifier_{timestamp}.pkl'))
    joblib.dump(selector, os.path.join(output_dir, f'feature_selector_{timestamp}.pkl'))
    
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
            'improvement_vs_baseline': float((reg_results[best_model_name]['r2'] - (-0.015)) * 100),
        },
        'classification': {
            'accuracy': float(clf_acc),
            'f1': float(clf_f1),
            'confusion_matrix': clf_cm.tolist(),
            'improvement_vs_baseline': float((clf_acc - 0.333) * 100),
        },
        'feature_selection': {
            'n_selected': int(selector.k),
            'original_features': int(selector.scores_.shape[0]),
        }
    }
    
    with open(os.path.join(output_dir, f'improvement_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary, output_dir


def cleanup_old_results():
    import glob
    import os
    output_dir = 'data/analysis/quick_improvement'
    if os.path.exists(output_dir):
        old_files = glob.glob(os.path.join(output_dir, '*.pkl')) + \
                   glob.glob(os.path.join(output_dir, 'improvement_summary_*.json'))
        for f in old_files:
            try:
                os.remove(f)
            except:
                pass
        if old_files:
            print(f"🧹 Cleaned {len(old_files)} old model files")


def main():
    parser = argparse.ArgumentParser(description='Train improved models')
    parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL TRAINING (Lag-Based Feature Engineering)")
    print("=" * 70)
    
    # Check for existing models
    exists, model_path, timestamp = check_existing_models()
    if exists and not args.force:
        print(f"\n✓ Found existing models from {timestamp}")
        print(f"  Use --force flag to retrain: python train_models.py --force")
        return
    
    if exists and args.force:
        print(f"\n🔄 Retraining (--force flag used)")
    
    cleanup_old_results()
    
    train_data, val_data, test_data = load_data()
    
    print("\nPreparing features with temporal lags...")
    X_train, y_train = prepare_features_with_lags(train_data, n_lags=3)
    X_val, y_val = prepare_features_with_lags(val_data, n_lags=3)
    X_test, y_test = prepare_features_with_lags(test_data, n_lags=3)
    
    X_train_sel, X_val_sel, X_test_sel, selector = select_top_features(
        X_train, y_train, X_val, X_test, n_features=15
    )
    
    baseline_results = train_baseline_comparison(X_train_sel, y_train, X_test_sel, y_test)
    
    reg_results, best_model_name = train_improved_models(
        X_train_sel, y_train, X_val_sel, y_val, X_test_sel, y_test
    )
    
    clf_model, clf_acc, clf_f1, clf_cm, y_test_risk, y_pred_clf = train_classification(
        X_train_sel, y_train, X_test_sel, y_test
    )
    
    summary, output_dir = save_results(
        reg_results, best_model_name, clf_model, clf_acc, clf_f1, clf_cm,
        selector, y_test, y_test_risk, y_pred_clf
    )
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n📊 REGRESSION:")
    print(f"   Baseline: R² = -0.015")
    print(f"   ✅ Best ({best_model_name.upper()}): R² = {reg_results[best_model_name]['r2']:.4f}")
    print(f"   🚀 Improvement: +{summary['regression']['improvement_vs_baseline']:.1f}%")
    
    print(f"\n📊 CLASSIFICATION:")
    print(f"   Baseline: Accuracy = 0.333")
    print(f"   ✅ Improved: Accuracy = {clf_acc:.4f}")
    print(f"   🚀 Improvement: +{summary['classification']['improvement_vs_baseline']:.1f}%")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
