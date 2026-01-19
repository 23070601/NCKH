"""
Evaluation utilities for model comparison
"""
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with MSE, RMSE, MAE, MAPE, R²
    """
    # Handle edge cases
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            'MSE': float('nan'),
            'RMSE': float('nan'),
            'MAE': float('nan'),
            'MAPE': float('nan'),
            'R2': float('nan')
        }
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE (handle division by zero)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # R² score
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a PyTorch model and return predictions.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (y_true, y_pred) arrays
    """
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass
            if hasattr(batch, 'edge_index'):
                out = model(batch.x, batch.edge_index, batch.edge_weight)
            else:
                out = model(batch.x)
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    return y_true, y_pred


def evaluate_sklearn_model(
    model,
    X: torch.Tensor,
    y: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a sklearn model.
    
    Args:
        model: Sklearn model with predict method
        X: Input features
        y: Target values
        
    Returns:
        Tuple of (y_true, y_pred) arrays
    """
    if isinstance(y, torch.Tensor):
        y_true = y.detach().cpu().numpy().flatten()
    else:
        y_true = y.flatten()
    
    y_pred = model.predict(X).flatten()
    
    return y_true, y_pred


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    n_samples: int = 100
) -> plt.Figure:
    """
    Plot true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        n_samples: Number of samples to plot
        
    Returns:
        Matplotlib figure
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Subsample if needed
    if len(y_true) > n_samples:
        indices = np.random.choice(len(y_true), n_samples, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(y_true_plot, y_pred_plot, alpha=0.5, s=20)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title(f'{model_name} - Predictions vs True Values')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true_plot - y_pred_plot
    axes[1].scatter(y_pred_plot, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predictions')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{model_name} - Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_time_series_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    stock_idx: int = 0,
    n_points: int = 100
) -> plt.Figure:
    """
    Plot time series predictions for a specific stock.
    
    Args:
        y_true: True values (timesteps, stocks)
        y_pred: Predicted values (timesteps, stocks)
        model_name: Name of the model
        stock_idx: Index of stock to plot
        n_points: Number of time points to plot
        
    Returns:
        Matplotlib figure
    """
    if len(y_true.shape) == 1:
        # Flatten case - reshape assuming batch format
        n_stocks = 98  # Default
        y_true = y_true.reshape(-1, n_stocks)
        y_pred = y_pred.reshape(-1, n_stocks)
    
    # Get data for specific stock
    true_series = y_true[:n_points, stock_idx]
    pred_series = y_pred[:n_points, stock_idx]
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    x = np.arange(len(true_series))
    ax.plot(x, true_series, label='True', linewidth=2, alpha=0.8)
    ax.plot(x, pred_series, label='Predicted', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.set_title(f'{model_name} - Time Series Prediction (Stock {stock_idx})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_models(
    results: Dict[str, Dict[str, float]]
) -> plt.Figure:
    """
    Compare multiple models using bar charts.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        
    Returns:
        Matplotlib figure
    """
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, float('nan')) for model in model_names]
        
        axes[i].bar(model_names, values, alpha=0.7)
        axes[i].set_title(f'{metric} Comparison', fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for j, v in enumerate(values):
            if not np.isnan(v):
                axes[i].text(j, v, f'{v:.4f}', ha='center', va='bottom')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    return fig


def print_metrics_table(results: Dict[str, Dict[str, float]]):
    """
    Print metrics in a formatted table.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
    """
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
    
    # Header
    print("\\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Column widths
    name_width = max(len(name) for name in results.keys()) + 2
    metric_width = 12
    
    # Header row
    header = f"{'Model':<{name_width}}"
    for metric in metrics:
        header += f"{metric:>{metric_width}}"
    print(header)
    print("-"*80)
    
    # Data rows
    for model_name, model_results in results.items():
        row = f"{model_name:<{name_width}}"
        for metric in metrics:
            value = model_results.get(metric, float('nan'))
            if np.isnan(value):
                row += f"{'N/A':>{metric_width}}"
            else:
                row += f"{value:>{metric_width}.6f}"
        print(row)
    
    print("="*80 + "\\n")


if __name__ == "__main__":
    # Test evaluation utilities
    
    # Generate dummy predictions
    n_samples = 1000
    y_true = np.random.randn(n_samples, 1)
    y_pred = y_true + np.random.randn(n_samples, 1) * 0.1  # Add some noise
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Plot predictions
    fig = plot_predictions(y_true, y_pred, "Test Model")
    plt.savefig('test_predictions.png')
    print("\\nPrediction plot saved to 'test_predictions.png'")
    
    # Test model comparison
    results = {
        'ARIMA': {'MSE': 0.0123, 'RMSE': 0.1109, 'MAE': 0.0891, 'MAPE': 5.23, 'R2': 0.75},
        'Random Forest': {'MSE': 0.0098, 'RMSE': 0.0990, 'MAE': 0.0756, 'MAPE': 4.12, 'R2': 0.82},
        'LSTM': {'MSE': 0.0085, 'RMSE': 0.0922, 'MAE': 0.0701, 'MAPE': 3.87, 'R2': 0.85},
        'GRU': {'MSE': 0.0079, 'RMSE': 0.0889, 'MAE': 0.0678, 'MAPE': 3.65, 'R2': 0.87}
    }
    
    print_metrics_table(results)
    
    fig = compare_models(results)
    plt.savefig('test_comparison.png')
    print("Comparison plot saved to 'test_comparison.png'")
