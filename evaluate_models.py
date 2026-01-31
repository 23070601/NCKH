import os
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def check_existing_evaluation(output_dir: Path = Path('data/analysis/evaluation_improved_lag')):
    """Check if evaluation results already exist."""
    metrics_file = output_dir / 'metrics.json'
    if metrics_file.exists():
        return True, str(metrics_file)
    return False, None


def find_predictions_file(analysis_dir: Path, input_arg: str = None) -> Path:
    if input_arg:
        return Path(input_arg)
    
    improved_lag_files = sorted(analysis_dir.glob("predictions_improved_lag_*.csv"))
    if improved_lag_files:
        return improved_lag_files[-1]
    
    all_pred_files = sorted(analysis_dir.glob("predictions_*.csv"))
    if all_pred_files:
        return all_pred_files[-1]
    
    raise FileNotFoundError("No predictions CSV file found")


def find_model_files(analysis_dir: Path) -> tuple[str, str]:
    regressor_files = sorted(glob.glob(str(analysis_dir / "improved_regressor_*.pkl")))
    classifier_files = sorted(glob.glob(str(analysis_dir / "improved_classifier_*.pkl")))
    
    if not regressor_files or not classifier_files:
        raise FileNotFoundError("Model files not found in quick_improvement directory")
    
    return regressor_files[-1], classifier_files[-1]


def evaluate_models(predictions_file: Path, output_dir: Path):
    print(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    print(f"Loaded {len(df)} samples")
    
    if 'Future_Vol_5D' not in df.columns or 'Pred_Vol' not in df.columns:
        raise ValueError("Future_Vol_5D or Pred_Vol column not found")
    
    y_true = df['Future_Vol_5D'].values
    y_pred = df['Pred_Vol'].values
    
    p33 = np.percentile(y_true, 33.33)
    p67 = np.percentile(y_true, 66.67)
    
    y_true_class = np.array([0 if v <= p33 else (1 if v <= p67 else 2) for v in y_true])
    y_pred_class = np.array([0 if v <= p33 else (1 if v <= p67 else 2) for v in y_pred])
    
    reg_r2 = r2_score(y_true, y_pred)
    reg_mae = np.mean(np.abs(y_true - y_pred))
    
    clf_acc = accuracy_score(y_true_class, y_pred_class)
    clf_precision = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    clf_recall = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    clf_f1 = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true_class, y_pred_class)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        'regression': {
            'r2': float(reg_r2),
            'mae': float(reg_mae),
        },
        'classification': {
            'accuracy': float(clf_acc),
            'precision': float(clf_precision),
            'recall': float(clf_recall),
            'f1': float(clf_f1),
            'confusion_matrix': cm.tolist(),
        }
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    report = classification_report(y_true_class, y_pred_class, zero_division=0)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Risk Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=100)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_indices = np.argsort(y_pred)
    ax.scatter(range(len(y_true)), y_true[sorted_indices], alpha=0.5, label='Actual', s=20)
    ax.scatter(range(len(y_pred)), y_pred[sorted_indices], alpha=0.5, label='Predicted', s=20)
    ax.set_xlabel('Samples (sorted by prediction)')
    ax.set_ylabel('Volatility')
    ax.set_title('Actual vs Predicted Volatility')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration.png', dpi=100)
    plt.close()
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\n📊 REGRESSION:")
    print(f"   R²:  {reg_r2:.4f}")
    print(f"   MAE: {reg_mae:.6f}")
    
    print(f"\n📊 CLASSIFICATION:")
    print(f"   Accuracy:  {clf_acc:.4f}")
    print(f"   Precision: {clf_precision:.4f}")
    print(f"   Recall:    {clf_recall:.4f}")
    print(f"   F1:        {clf_f1:.4f}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - metrics.json")
    print(f"  - classification_report.txt")
    print(f"  - confusion_matrix.png")
    print(f"  - calibration.png")
    print("=" * 70)


def cleanup_old_results():
    from pathlib import Path
    output_dir = Path('data/analysis/evaluation_improved_lag')
    if output_dir.exists():
        old_files = list(output_dir.glob('*.json')) + list(output_dir.glob('*.txt')) + list(output_dir.glob('*.png'))
        for f in old_files:
            try:
                os.remove(f)
            except:
                pass
        if old_files:
            print(f"🧹 Cleaned {len(old_files)} old evaluation files")


def main():
    parser = argparse.ArgumentParser(description='Evaluate improved models')
    parser.add_argument('--input', type=str, help='Input predictions CSV file')
    parser.add_argument('--output-dir', type=str, default='data/analysis/evaluation_improved_lag',
                       help='Output directory for evaluation results')
    parser.add_argument('--force', action='store_true', help='Force re-evaluation even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Check for existing evaluation
    exists, metrics_file = check_existing_evaluation(output_dir)
    if exists and not args.force:
        print("=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        print(f"\n✓ Found existing evaluation results")
        print(f"  File: {metrics_file}")
        print(f"  Use --force flag to re-evaluate: python evaluate_models.py --force")
        return
    
    if exists and args.force:
        print("🔄 Re-evaluating (--force flag used)\n")
    
    cleanup_old_results()
    
    analysis_dir = Path('data/analysis')
    predictions_file = find_predictions_file(analysis_dir, args.input)
    
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    evaluate_models(predictions_file, output_dir)


if __name__ == '__main__':
    main()
