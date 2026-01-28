# Enhanced Dataset Features

## Overview
This document describes the enhanced features added to improve stock volatility and risk prediction for Vietnamese FDI enterprises.

## New Modules

### 1. Macroeconomic Data Collection (`src/macro_data.py`)

Collects market-level indicators that influence stock volatility:

**Features:**
- **VN-Index**: Vietnam stock market benchmark index
  - Daily close values
  - Daily returns
  - 20-day rolling volatility
  
- **Exchange Rates**: USD/VND
  - Daily exchange rate
  - Daily percentage change
  
- **Interest Rates**:
  - SBV policy rate (quarterly updates)
  - Interbank overnight rate (daily)
  
- **Inflation**:
  - Consumer Price Index (CPI) - monthly
  - Year-over-Year inflation rate

**Usage:**
```python
from src.macro_data import MacroDataCollector

collector = MacroDataCollector(
    start_date='2022-01-01',
    end_date='2024-12-31'
)
macro_df = collector.collect_all()
collector.save('data/features/macro_data.csv')
```

### 2. Advanced Risk Metrics (`src/risk_metrics.py`)

Implements comprehensive risk measurement tools:

**Metrics:**

- **Multi-Horizon Volatility**: 5-day, 20-day, 60-day rolling windows
- **Value at Risk (VaR)**: Maximum loss at 95% and 99% confidence
  - Historical, Parametric, and Cornish-Fisher methods
- **Conditional VaR (CVaR)**: Expected tail loss beyond VaR
- **Downside Risk**: Semi-deviation of negative returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Beta**: Systematic risk relative to market

**Usage:**
```python
from src.risk_metrics import RiskMetrics

# Calculate all metrics
metrics = RiskMetrics.calculate_all_metrics(
    returns=stock_returns,
    prices=stock_prices,
    market_returns=vn_index_returns
)

# Individual metrics
var_95 = RiskMetrics.value_at_risk(returns, confidence_level=0.95)
cvar_95 = RiskMetrics.conditional_var(returns, confidence_level=0.95)
sharpe = RiskMetrics.sharpe_ratio(returns)
```

**Risk Classification:**
```python
from src.risk_metrics import calculate_risk_labels

# Create risk levels: 0=Low, 1=Medium, 2=High
risk_labels = calculate_risk_labels(
    volatility_values,
    method='percentile',  # or 'threshold', 'kmeans'
    n_classes=3
)
```

### 3. Hybrid GNN-LSTM Model (`src/models/hybrid_gnn_lstm.py`)

Combines temporal and graph learning for superior prediction:

**Architecture:**

1. **Temporal Encoder** (LSTM/GRU): Captures time-series patterns per stock
2. **Graph Encoder** (GCN/GAT/SAGE): Aggregates cross-stock dependencies
3. **Fusion Layer**: Combines temporal and graph features
   - Concatenation
   - Addition
   - Attention-based fusion
4. **Predictor Head**: Final regression/classification layer

**Single-Task Model:**
```python
from src.models import HybridGNNLSTM

model = HybridGNNLSTM(
    in_features=8,
    hidden_size=64,
    num_temporal_layers=2,
    num_graph_layers=2,
    rnn_type='lstm',
    gnn_type='gcn',
    fusion_method='concat',
    output_size=1
)

predictions = model(x, edge_index, edge_weight)
```

**Multi-Task Model:**
```python
from src.models import MultiTaskHybridGNN

model = MultiTaskHybridGNN(
    in_features=8,
    hidden_size=64,
    num_risk_classes=3
)

output = model(x, edge_index, edge_weight)
volatility_pred = output['volatility']
risk_class = output['risk_class']
```

### 4. Enhanced Fundamental Features (`src/data_utils.py`)

Expanded from 7 to 17+ fundamental indicators:

**Valuation:**
- P/E ratio (Price-to-Earnings)
- P/B ratio (Price-to-Book)
- P/S ratio (Price-to-Sales)
- PEG ratio (Price-to-Earnings-Growth)

**Profitability:**
- ROE (Return on Equity)
- ROA (Return on Assets)
- Profit Margin
- Operating Margin

**Leverage:**
- Debt-to-Equity ratio
- Debt-to-Assets ratio
- Current Ratio
- Quick Ratio

**Growth:**
- Revenue Growth
- EPS Growth

**Market:**
- Market Capitalization
- Beta (systematic risk)
- Dividend Yield
- Average Trading Volume

**Usage:**
```python
from src.data_utils import collect_fundamentals_for_tickers

fundamentals_df = collect_fundamentals_for_tickers(
    tickers=['VNM', 'SAB', 'MSN'],
    use_api=True,  # Use vnstock3 API
    save_path='data/features/fundamentals.csv'
)
```

### 5. Enhanced Datasets (`src/datasets/EnhancedDataset.py`)

**EnhancedVNStocksDataset:**
- Integrates technical + macro + fundamental features
- Configurable feature inclusion
- Backward compatible with base dataset

**EnhancedVolatilityDataset:**
- Multi-target prediction:
  - Future volatility
  - VaR and CVaR
  - Risk classification labels
- Supports multi-task learning

**Usage:**
```python
from src.datasets import EnhancedVolatilityDataset

dataset = EnhancedVolatilityDataset(
    root='data/',
    past_window=25,
    future_window=5,
    volatility_window=20,
    include_macro=True,
    include_fundamentals=True,
    calculate_var=True,
    risk_labels=True,
    num_risk_classes=3
)

sample = dataset[0]
# Access features
x = sample.x  # Includes technical + macro + fundamentals
volatility = sample.y  # Future volatility
var = sample.var  # Value at Risk
risk_class = sample.risk_class  # Risk level (0/1/2)
```

## Integration Workflow

### Step 1: Collect Macro Data
```python
from src.macro_data import MacroDataCollector

collector = MacroDataCollector('2022-01-01', '2024-12-31')
macro_df = collector.collect_all()
collector.save('data/features/macro_data.csv')
```

### Step 2: Collect Fundamentals
```python
from src.data_utils import collect_fundamentals_for_tickers
import pandas as pd

tickers = pd.read_csv('data/features/tickers.csv')['ticker'].tolist()
fundamentals = collect_fundamentals_for_tickers(
    tickers,
    use_api=False,  # Use simulated data for demo
    save_path='data/features/fundamentals.csv'
)
```

### Step 3: Enrich Stock Data
```python
from src.macro_data import integrate_macro_with_stock_data

enriched_df = integrate_macro_with_stock_data(
    stock_data_path='data/processed/values.csv',
    macro_data_path='data/features/macro_data.csv',
    output_path='data/processed/values_enriched.csv'
)
```

### Step 4: Create Enhanced Dataset
```python
from src.datasets import EnhancedVolatilityDataset

dataset = EnhancedVolatilityDataset(
    root='data/',
    values_file_name='values_enriched.csv',
    fundamentals_file_name='fundamentals.csv',
    include_macro=True,
    include_fundamentals=True
)
```

### Step 5: Train Hybrid Model
```python
from src.models import MultiTaskHybridGNN
from src.utils import train
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim

# Create data loaders
train_loader = DataLoader(dataset[:600], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset[600:], batch_size=16, shuffle=False)

# Create model
model = MultiTaskHybridGNN(
    in_features=dataset[0].x.shape[1],  # Auto-detect feature count
    hidden_size=64,
    num_risk_classes=3,
    rnn_type='lstm',
    gnn_type='gat',
    fusion_method='attention'
)

# Multi-task loss
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, target):
        vol_loss = self.mse(pred['volatility'], target.y)
        risk_loss = self.ce(pred['risk_logits'], target.risk_class)
        return vol_loss + 0.5 * risk_loss

# Train
history = train(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    criterion=MultiTaskLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=50
)
```

## Feature Counts

| Dataset Type | Technical | Macro | Fundamentals | Total |
|--------------|-----------|-------|--------------|-------|
| Base | 8 | 0 | 0 | 8 |
| + Macro | 8 | 9 | 0 | 17 |
| + Fundamentals | 8 | 0 | 17 | 25 |
| Enhanced (All) | 8 | 9 | 17 | **34** |

## Model Comparison

| Model | Parameters | Features | Graph | Temporal | Multi-Task |
|-------|------------|----------|-------|----------|------------|
| ARIMA | ~10 | 1 | ✗ | ✓ | ✗ |
| Random Forest | ~100K | All | ✗ | ✗ | ✗ |
| LSTM | ~50K | All | ✗ | ✓ | ✗ |
| GRU | ~40K | All | ✗ | ✓ | ✗ |
| **HybridGNN** | **~150K** | **All** | **✓** | **✓** | **✓** |

## Expected Improvements

Based on the enhanced features and hybrid architecture:

1. **Volatility Prediction**: 10-15% reduction in RMSE
2. **Risk Classification**: 15-20% improvement in accuracy
3. **Tail Risk Capture**: Better VaR/CVaR estimation
4. **Cross-Stock Dependencies**: Improved correlation modeling
5. **Macro Sensitivity**: Better capture of market-wide shocks

## References

- Macroeconomic indicators: State Bank of Vietnam (SBV)
- Fundamental data: vnstock3 API
- Risk metrics: Industry-standard measures (Basel II/III)
- Hybrid architecture: Graph Neural Networks + Temporal Modeling

## Next Steps

1. Validate with real API data (vnstock3)
2. Hyperparameter tuning for hybrid model
3. Ensemble methods combining multiple models
4. Real-time prediction pipeline
5. Portfolio optimization using risk predictions
