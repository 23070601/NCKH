"""
Volatility and Risk Level Prediction - Training Pipeline
Trains models to predict future volatility and risk classifications
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datasets import EnhancedVolatilityDataset
from models import (
    ARIMAModel,
    RandomForestModel,
    LSTMModel,
    GRUModel,
    MultiTaskHybridGNN
)
from risk_metrics import RiskMetrics, calculate_risk_labels
from utils.train import train
from utils.evaluate import (
    evaluate_model,
    evaluate_sklearn_model,
    calculate_metrics,
    print_metrics_table,
    compare_models,
    plot_predictions
)


def define_risk_labels(volatilities, method='percentile', n_classes=3):
    """
    Define risk labels based on volatility distribution.
    
    Args:
        volatilities: Array of volatility values
        method: 'percentile', 'threshold', or 'kmeans'
        n_classes: Number of risk classes (3 = Low/Medium/High)
    
    Returns:
        Risk labels and thresholds
    """
    print("="*80)
    print("DEFINING RISK LABELS")
    print("="*80)
    
    labels = calculate_risk_labels(volatilities, method=method, n_classes=n_classes)
    
    # Calculate thresholds
    if method == 'percentile':
        percentiles = [33.33, 66.67] if n_classes == 3 else np.linspace(0, 100, n_classes+1)[1:-1]
        thresholds = np.percentile(volatilities, percentiles)
    elif method == 'threshold':
        thresholds = [0.15, 0.30] if n_classes == 3 else np.linspace(0.1, 0.5, n_classes-1)
    else:
        thresholds = None
    
    # Print distribution
    print(f"\nMethod: {method}")
    print(f"Number of classes: {n_classes}")
    
    if thresholds is not None:
        print(f"\nThresholds:")
        for i, t in enumerate(thresholds):
            print(f"  Class {i} -> {i+1}: {t:.4f}")
    
    print(f"\nLabel Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ['Low', 'Medium', 'High'] if n_classes == 3 else [f'Level_{i}' for i in range(n_classes)]
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        print(f"  {label_names[label]:10s}: {count:6d} ({pct:5.1f}%)")
    
    return labels, thresholds


def create_datasets(
    root='data/',
    past_window=25,
    future_window=5,
    volatility_window=20,
    train_split=0.7,
    val_split=0.15
):
    """
    Create train/validation/test datasets with risk labels.
    """
    print("\n" + "="*80)
    print("CREATING ENHANCED DATASETS")
    print("="*80)
    
    # Create full dataset
    dataset = EnhancedVolatilityDataset(
        root=root,
        past_window=past_window,
        future_window=future_window,
        volatility_window=volatility_window,
        include_macro=True,
        include_fundamentals=True,
        calculate_var=True,
        risk_labels=True,
        num_risk_classes=3,
        force_reload=False
    )
    
    print(f"\n✓ Dataset created: {len(dataset)} samples")
    
    # Get sample to check dimensions
    sample = dataset[0]
    print(f"  Feature dimensions: {sample.x.shape}")
    print(f"  Number of features: {sample.x.shape[1]}")
    print(f"  Time steps: {sample.x.shape[2]}")
    print(f"  Number of stocks: {sample.x.shape[0]}")
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    train_dataset = dataset[:n_train]
    val_dataset = dataset[n_train:n_train + n_val]
    test_dataset = dataset[n_train + n_val:]
    
    print(f"\n✓ Data split:")
    print(f"  Train: {len(train_dataset)} samples ({train_split*100:.0f}%)")
    print(f"  Val:   {len(val_dataset)} samples ({val_split*100:.0f}%)")
    print(f"  Test:  {len(test_dataset)} samples ({(1-train_split-val_split)*100:.0f}%)")
    
    return train_dataset, val_dataset, test_dataset, sample.x.shape[1]


def train_baseline_models(train_data, test_data, device='cpu'):
    """
    Train baseline models: ARIMA, Random Forest, LSTM, GRU.
    """
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    
    results = {}
    
    # Prepare data for sklearn models (flatten temporal dimension)
    X_train = torch.stack([data.x for data in train_data])
    y_train = torch.stack([data.y for data in train_data])
    X_test = torch.stack([data.x for data in test_data])
    y_test = torch.stack([data.y for data in test_data])
    
    # 1. Random Forest
    print("\n[1/4] Training Random Forest...")
    rf_model = RandomForestModel(n_estimators=100, max_depth=20, random_state=42)
    rf_model.fit(X_train, y_train)
    y_true_rf, y_pred_rf = evaluate_sklearn_model(rf_model, X_test, y_test)
    results['Random Forest'] = calculate_metrics(y_true_rf, y_pred_rf)
    print(f"  ✓ RMSE: {results['Random Forest']['RMSE']:.6f}")
    
    # 2. LSTM
    print("\n[2/4] Training LSTM...")
    lstm_model = LSTMModel(
        in_features=train_data[0].x.shape[1],
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    history_lstm = train(
        model=lstm_model,
        optimizer=optim.Adam(lstm_model.parameters(), lr=0.001),
        criterion=nn.MSELoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=30,
        device=device,
        task_title="lstm_volatility",
        early_stopping_patience=5
    )
    
    y_true_lstm, y_pred_lstm = evaluate_model(lstm_model, test_loader, device)
    results['LSTM'] = calculate_metrics(y_true_lstm, y_pred_lstm)
    print(f"  ✓ Best RMSE: {np.sqrt(history_lstm['best_test_loss']):.6f}")
    
    # 3. GRU
    print("\n[3/4] Training GRU...")
    gru_model = GRUModel(
        in_features=train_data[0].x.shape[1],
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    history_gru = train(
        model=gru_model,
        optimizer=optim.Adam(gru_model.parameters(), lr=0.001),
        criterion=nn.MSELoss(),
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=30,
        device=device,
        task_title="gru_volatility",
        early_stopping_patience=5
    )
    
    y_true_gru, y_pred_gru = evaluate_model(gru_model, test_loader, device)
    results['GRU'] = calculate_metrics(y_true_gru, y_pred_gru)
    print(f"  ✓ Best RMSE: {np.sqrt(history_gru['best_test_loss']):.6f}")
    
    # 4. ARIMA (sample for speed - run on subset)
    print("\n[4/4] Training ARIMA (on subset for speed)...")
    print("  Note: ARIMA trains separately per stock, showing results for first stock")
    
    # Get data for one stock
    X_arima = X_train[0, 0, :, :].unsqueeze(0)  # First stock
    y_arima = y_train[0, 0].unsqueeze(0)
    
    arima_model = ARIMAModel(order=(2, 1, 1))
    arima_model.fit(X_arima)
    arima_pred = arima_model.predict(X_test[0, 0, :, :].unsqueeze(0), steps=1)
    
    # For full ARIMA evaluation, would need to iterate over all stocks
    results['ARIMA (sample)'] = {'RMSE': 'N/A (placeholder)', 'MAE': 'N/A'}
    print(f"  ✓ ARIMA trained (per-stock basis)")
    
    return results, {
        'lstm': lstm_model,
        'gru': gru_model,
        'rf': rf_model
    }


def train_hybrid_model(train_data, val_data, test_data, n_features, device='cpu'):
    """
    Train hybrid GNN-LSTM model with multi-task learning.
    """
    print("\n" + "="*80)
    print("TRAINING HYBRID GNN-LSTM MODEL")
    print("="*80)
    
    # Create model
    model = MultiTaskHybridGNN(
        in_features=n_features,
        hidden_size=64,
        num_risk_classes=3,
        num_temporal_layers=2,
        num_graph_layers=2,
        rnn_type='lstm',
        gnn_type='gat',
        fusion_method='attention',
        dropout=0.2
    ).to(device)
    
    print(f"\n✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    
    # Multi-task loss
    class MultiTaskLoss(nn.Module):
        def __init__(self, vol_weight=1.0, risk_weight=0.5):
            super().__init__()
            self.vol_weight = vol_weight
            self.risk_weight = risk_weight
            self.mse = nn.MSELoss()
            self.ce = nn.CrossEntropyLoss()
        
        def forward(self, pred, target):
            vol_loss = self.mse(pred['volatility'], target.y)
            
            # Risk classification loss (if labels available)
            if hasattr(target, 'risk_class'):
                risk_loss = self.ce(pred['risk_logits'], target.risk_class)
                return self.vol_weight * vol_loss + self.risk_weight * risk_loss
            else:
                return vol_loss
    
    criterion = MultiTaskLoss()
    
    # Train
    print("\n✓ Training started...")
    history = train(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        criterion=criterion,
        train_loader=train_loader,
        test_loader=val_loader,
        num_epochs=50,
        device=device,
        task_title="hybrid_gnn_lstm",
        early_stopping_patience=10
    )
    
    # Evaluate on test set
    print("\n✓ Evaluating on test set...")
    model.eval()
    
    all_vol_true = []
    all_vol_pred = []
    all_risk_true = []
    all_risk_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.edge_weight)
            
            all_vol_true.append(batch.y.cpu().numpy())
            all_vol_pred.append(output['volatility'].cpu().numpy())
            
            if hasattr(batch, 'risk_class'):
                all_risk_true.append(batch.risk_class.cpu().numpy())
                all_risk_pred.append(output['risk_class'].cpu().numpy())
    
    # Calculate metrics
    vol_true = np.concatenate(all_vol_true)
    vol_pred = np.concatenate(all_vol_pred)
    vol_metrics = calculate_metrics(vol_true, vol_pred)
    
    results = {
        'volatility_metrics': vol_metrics,
        'best_epoch': history['best_epoch'],
        'best_loss': history['best_test_loss']
    }
    
    # Risk classification metrics
    if len(all_risk_true) > 0:
        risk_true = np.concatenate(all_risk_true)
        risk_pred = np.concatenate(all_risk_pred)
        
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(risk_true, risk_pred)
        
        results['risk_accuracy'] = accuracy
        results['risk_report'] = classification_report(
            risk_true,
            risk_pred,
            target_names=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        print(f"\n✓ Risk Classification Accuracy: {accuracy:.4f}")
    
    print(f"✓ Volatility RMSE: {vol_metrics['RMSE']:.6f}")
    
    return model, results


def save_results(baseline_results, hybrid_results, output_dir='data/analysis'):
    """
    Save training results and generate comparison plots.
    """
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine results
    all_results = {**baseline_results}
    all_results['Hybrid GNN-LSTM'] = hybrid_results['volatility_metrics']
    
    # Print comparison table
    print("\n✓ Model Comparison:")
    print_metrics_table(all_results)
    
    # Save to JSON
    import json
    results_file = os.path.join(output_dir, 'volatility_prediction_results.json')
    
    # Convert to serializable format
    serializable_results = {}
    for model, metrics in all_results.items():
        serializable_results[model] = {
            k: float(v) if isinstance(v, (np.floating, float)) else str(v)
            for k, v in metrics.items()
        }
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Generate comparison plot
    fig = compare_models(all_results)
    plot_file = os.path.join(output_dir, 'model_comparison.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to {plot_file}")
    
    # Save risk classification report if available
    if 'risk_report' in hybrid_results:
        risk_file = os.path.join(output_dir, 'risk_classification_report.txt')
        with open(risk_file, 'w') as f:
            f.write("RISK CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Accuracy: {hybrid_results['risk_accuracy']:.4f}\n\n")
            f.write(hybrid_results['risk_report'])
        
        print(f"✓ Risk report saved to {risk_file}")
    
    plt.close('all')


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*80)
    print("VOLATILITY AND RISK PREDICTION - TRAINING PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Step 1: Create datasets with risk labels
    train_data, val_data, test_data, n_features = create_datasets(
        root='data/',
        past_window=25,
        future_window=5,
        volatility_window=20,
        train_split=0.7,
        val_split=0.15
    )
    
    # Step 2: Train baseline models
    baseline_results, baseline_models = train_baseline_models(
        train_data,
        test_data,
        device=device
    )
    
    # Step 3: Train hybrid model
    hybrid_model, hybrid_results = train_hybrid_model(
        train_data,
        val_data,
        test_data,
        n_features,
        device=device
    )
    
    # Step 4: Save results
    save_results(baseline_results, hybrid_results)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ All models trained successfully")
    print("✓ Results saved to data/analysis/")
    print("\nNext steps:")
    print("  1. Review model comparison in data/analysis/model_comparison.png")
    print("  2. Analyze risk classification report")
    print("  3. Fine-tune best performing model")
    print("  4. Generate predictions for portfolio optimization")


if __name__ == "__main__":
    main()
