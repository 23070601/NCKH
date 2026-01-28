"""
Quick Start Guide: Using Enhanced Features
==========================================

This script demonstrates how to use the new enhanced features
for stock volatility and risk prediction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Step 1: Collect Macroeconomic Data
print("="*80)
print("STEP 1: Collecting Macroeconomic Data")
print("="*80)

from src.macro_data import MacroDataCollector

collector = MacroDataCollector(
    start_date='2022-01-01',
    end_date='2024-12-31'
)

macro_df = collector.collect_all()

# Save macro data
os.makedirs('data/features', exist_ok=True)
collector.save('data/features/macro_data.csv')

print("\n" + "="*80)
print("STEP 2: Collecting Fundamental Features")
print("="*80)

from src.data_utils import collect_fundamentals_for_tickers
import pandas as pd

# Load tickers
tickers_df = pd.read_csv('data/features/tickers.csv')
tickers = tickers_df['ticker'].tolist()

# Collect fundamentals (using simulated data for demo)
fundamentals_df = collect_fundamentals_for_tickers(
    tickers=tickers,
    use_api=False,  # Set to True to use vnstock3 API
    save_path='data/features/fundamentals.csv'
)

print("\n" + "="*80)
print("STEP 3: Integrating Macro Data with Stock Data")
print("="*80)

from src.macro_data import integrate_macro_with_stock_data

# Check if values.csv exists in processed directory
values_path = 'data/processed/values.csv'
if not os.path.exists(values_path):
    print(f"Warning: {values_path} not found. Run data collection first.")
    print("Skipping integration step.")
else:
    enriched_df = integrate_macro_with_stock_data(
        stock_data_path=values_path,
        macro_data_path='data/features/macro_data.csv',
        output_path='data/processed/values_enriched.csv'
    )

print("\n" + "="*80)
print("STEP 4: Testing Risk Metrics")
print("="*80)

from src.risk_metrics import RiskMetrics, calculate_risk_labels
import numpy as np

# Generate sample returns for demonstration
np.random.seed(42)
sample_returns = np.random.standard_t(df=5, size=252) * 0.02
sample_prices = 100 * np.exp(np.cumsum(sample_returns))

# Calculate risk metrics
metrics = RiskMetrics.calculate_all_metrics(
    returns=sample_returns,
    prices=sample_prices
)

print("\nRisk Metrics:")
for metric, value in metrics.items():
    print(f"  {metric:20s}: {value:10.4f}")

# Test risk classification
volatilities = np.random.lognormal(np.log(0.2), 0.5, 1000)
risk_labels = calculate_risk_labels(
    volatilities,
    method='percentile',
    n_classes=3
)

print(f"\nRisk Labels Distribution:")
unique, counts = np.unique(risk_labels, return_counts=True)
for label, count in zip(unique, counts):
    risk_name = ['Low', 'Medium', 'High'][label]
    print(f"  {risk_name}: {count} ({count/len(risk_labels)*100:.1f}%)")

print("\n" + "="*80)
print("STEP 5: Loading Enhanced Dataset")
print("="*80)

# This step requires processed data - check if enriched file exists
if os.path.exists('data/processed/values_enriched.csv'):
    from src.datasets import EnhancedVolatilityDataset
    
    try:
        dataset = EnhancedVolatilityDataset(
            root='data/',
            values_file_name='values_enriched.csv',
            fundamentals_file_name='fundamentals.csv',
            past_window=25,
            future_window=5,
            volatility_window=20,
            include_macro=True,
            include_fundamentals=True,
            calculate_var=True,
            risk_labels=True,
            num_risk_classes=3,
            force_reload=True
        )
        
        print(f"\n✓ Enhanced Dataset Created")
        print(f"  Total samples: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Feature dimensions: {sample.x.shape}")
            print(f"  Target shape: {sample.y.shape}")
            print(f"  Has risk labels: {hasattr(sample, 'risk_class')}")
            print(f"  Has VaR/CVaR: {hasattr(sample, 'var')}")
    
    except Exception as e:
        print(f"\nNote: Could not create enhanced dataset: {e}")
        print("This is normal if you haven't run data collection yet.")
else:
    print("\nNote: Enriched data file not found.")
    print("Run data collection notebooks first to generate processed data.")

print("\n" + "="*80)
print("STEP 6: Model Architecture Preview")
print("="*80)

from src.models import HybridGNNLSTM, MultiTaskHybridGNN
import torch

# Create a small test model
print("\nCreating Hybrid GNN-LSTM Model...")

model = HybridGNNLSTM(
    in_features=34,  # Technical + Macro + Fundamentals
    hidden_size=64,
    num_temporal_layers=2,
    num_graph_layers=2,
    rnn_type='lstm',
    gnn_type='gcn',
    fusion_method='concat',
    output_size=1
)

params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Single-Task Model created")
print(f"  Parameters: {params:,}")

# Create multi-task model
mt_model = MultiTaskHybridGNN(
    in_features=34,
    hidden_size=64,
    num_risk_classes=3,
    rnn_type='gru',
    gnn_type='gat',
    fusion_method='attention'
)

mt_params = sum(p.numel() for p in mt_model.parameters())
print(f"\n  ✓ Multi-Task Model created")
print(f"  Parameters: {mt_params:,}")
print(f"  Tasks: Volatility Prediction + Risk Classification")

# Test forward pass with dummy data
dummy_x = torch.randn(10, 34, 25)  # 10 nodes, 34 features, 25 timesteps
dummy_edge_index = torch.randint(0, 10, (2, 20))
dummy_edge_weight = torch.rand(20)

output = mt_model(dummy_x, dummy_edge_index, dummy_edge_weight)
print(f"\n  ✓ Forward pass successful")
print(f"  Volatility output: {output['volatility'].shape}")
print(f"  Risk logits: {output['risk_logits'].shape}")
print(f"  Risk classes: {output['risk_class'].shape}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n✓ All enhanced features successfully implemented:")
print("  1. Macroeconomic data collection")
print("  2. Fundamental features (17 metrics)")
print("  3. Advanced risk metrics (VaR, CVaR, Sharpe, etc.)")
print("  4. Hybrid GNN-LSTM model architecture")
print("  5. Enhanced datasets with multi-task targets")
print("  6. Risk classification labels")
print("\n📖 See ENHANCED_FEATURES.md for detailed documentation")
print("📖 See README.md for updated usage examples")
print("\nNext Steps:")
print("  1. Run data collection notebooks to generate processed data")
print("  2. Train models using the enhanced features")
print("  3. Compare performance with baseline models")
print("  4. Analyze risk predictions and classifications")
print("\n" + "="*80)
