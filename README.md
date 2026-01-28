# Vietnamese FDI Stock Volatility Prediction

## Project Structure

```
NCKH/
├── README.md
├── requirements.txt
├── collect_data.py          # Data collection entry point
│
├── data/
│   ├── raw/                        # Raw inputs for datasets
│   │   ├── values.csv              # Stock price & features (75,754 rows × 9 cols)
│   │   ├── adj.npy                 # Adjacency matrix (98×98)
│   │   └── fdi_stocks_list.csv     # List of FDI stocks
│   ├── processed/                  # Cached PyTorch Geometric artifacts
│   └── analysis/                   # Generated analysis outputs
│
├── notebooks/
│   ├── 0_data_collection.ipynb      # Data collection demo
│   ├── 1_data_preparation.ipynb     # Exploratory data analysis
│   ├── 2_data_preparation.ipynb     # PyTorch Geometric dataset creation
│   └── 3_model_comparison.ipynb     # Model training & comparison
│
└── src/
    ├── VNStocks.py          # Data collection class
    ├── data_utils.py        # Utility functions
    ├── macro_data.py        # Macroeconomic data collection (NEW)
    ├── risk_metrics.py      # Advanced risk metrics (NEW)
    │
    ├── datasets/
    │   ├── VNStocksDataset.py       # PyTorch Geometric datasets
    │   └── EnhancedDataset.py       # Enhanced datasets with macro/fundamentals (NEW)
    │
    ├── models/
    │   ├── arima.py                 # ARIMA baseline
    │   ├── random_forest.py         # Random Forest model
    │   ├── lstm.py                  # LSTM model
    │   ├── gru.py                   # GRU model
    │   └── hybrid_gnn_lstm.py       # Hybrid GNN-LSTM model (NEW)
    │
    └── utils/
        ├── train.py                 # Training utilities
        └── evaluate.py              # Evaluation metrics
```

## Enhanced Features (NEW)

### Macroeconomic Indicators
- **VN-Index**: Market benchmark (close, returns, volatility)
- **Exchange Rate**: USD/VND (rate, changes)
- **Interest Rates**: Policy rate, interbank rate
- **Inflation**: CPI, year-over-year inflation

### Fundamental Features (17 metrics)
- **Valuation**: P/E, P/B, P/S, PEG ratios
- **Profitability**: ROE, ROA, profit margin, operating margin
- **Leverage**: Debt-to-Equity, current ratio, quick ratio
- **Growth**: Revenue growth, EPS growth
- **Market**: Market cap, beta, dividend yield, volume

### Advanced Risk Metrics
- **Multi-Horizon Volatility**: 5-day, 20-day, 60-day
- **Value at Risk (VaR)**: 95% and 99% confidence
- **Conditional VaR (CVaR)**: Expected tail loss
- **Sharpe/Sortino Ratios**: Risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline

See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for detailed documentation.
```

## Dataset

### Stock Data
- **98 Vietnamese FDI stocks** (2022-01-01 to 2024-12-31)
- **773 trading days** of historical data
- **9 features** per stock:
- Raw files stored in `data/raw/values.csv` and `data/raw/adj.npy`


| Feature | Description | Formula |
|---------|-------------|---------|
| Close | Closing price | Raw price |
| NormClose | Normalized close | Close / First Close |
| DailyLogReturn | Daily log return | log(Close_t / Close_{t-1}) |

| ALR1W | 1-week annualized return | (Close_t / Close_{t-5})^(252/5) - 1 |
| ALR2W | 2-week annualized return | (Close_t / Close_{t-10})^(252/10) - 1 |
| ALR1M | 1-month annualized return | (Close_t / Close_{t-21})^(252/21) - 1 |
| ALR2M | 2-month annualized return | (Close_t / Close_{t-42})^(252/42) - 1 |
| RSI | Relative Strength Index | Technical momentum indicator |
| MACD | Moving Average Convergence Divergence | Trend-following indicator |


### Graph Structure
- **Adjacency matrix**: 98×98 correlation-based graph
- **52 edges** (correlation threshold: 0.1)
- **Graph density**: 0.54%
- **Purpose**: Capture relationships between stocks

## Models

| Group | Algorithm | Role | Implementation |
|-------|-----------|------|----------------|
| **Baseline** | ARIMA | Statistical benchmark | `src/models/arima.py` |
| **ML** | Random Forest | Non-linear baseline | `src/models/random_forest.py` |
| **DL** | LSTM | Temporal modeling | `src/models/lstm.py` |
| **DL** | GRU | Lighter than LSTM | `src/models/gru.py` |
| **Advanced** | **Hybrid GNN-LSTM** | **Graph + Temporal** | `src/models/hybrid_gnn_lstm.py` **(NEW)** |

### Model Details

#### ARIMA (AutoRegressive Integrated Moving Average)
- Classical statistical model for time series
- Separate model for each stock
- Order: (p=2, d=1, q=1)
- **Pros**: Interpretable, no training required
- **Cons**: Cannot capture non-linear relationships

#### Random Forest
- Ensemble of decision trees
- Treats each timestep independently
- 100 trees, max depth 20
- **Pros**: Non-linear, handles missing data
- **Cons**: Doesn't model temporal dependencies

#### LSTM (Long Short-Term Memory)
- Recurrent neural network
- 2 layers, 64 hidden units
- Dropout: 0.2
- **Pros**: Captures long-term dependencies
- **Cons**: More parameters, slower training

#### GRU (Gated Recurrent Unit)
- Lighter alternative to LSTM
- 2 layers, 64 hidden units
- ~25% fewer parameters than LSTM
- **Pros**: Faster training, similar performance
- **Cons**: Slightly less expressive than LSTM

#### Hybrid GNN-LSTM (NEW)
- **Combines temporal and graph modeling**
- **Architecture**:
  1. Temporal Encoder (LSTM/GRU): Process time series per stock
  2. Graph Encoder (GCN/GAT/SAGE): Aggregate cross-stock dependencies
  3. Fusion Layer: Combine temporal + graph features (concat/add/attention)
  4. Predictor: Final output for volatility/risk
- **Parameters**: ~150K (configurable)
- **Multi-Task**: Can predict volatility + risk classification simultaneously
- **Pros**: Captures both temporal patterns and market structure
- **Cons**: More complex, requires more data

## Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **MSE** | Mean Squared Error | Mean of squared differences |
| **RMSE** | Root Mean Squared Error | Square root of MSE |
| **MAE** | Mean Absolute Error | Mean of absolute differences |
| **MAPE** | Mean Absolute Percentage Error | Percentage error |
| **R²** | Coefficient of Determination | Proportion of variance explained |

## Usage Examples

### Load Enhanced Dataset (NEW)

```python
from src.datasets import EnhancedVolatilityDataset

# Enhanced dataset with all features
dataset = EnhancedVolatilityDataset(
    root='data/',
    values_file_name='values_enriched.csv',
    past_window=25,
    future_window=5,
    volatility_window=20,
    include_macro=True,           # Include macro features
    include_fundamentals=True,    # Include fundamental features
    calculate_var=True,            # Calculate VaR and CVaR
    risk_labels=True,              # Generate risk class labels
    num_risk_classes=3             # Low/Medium/High
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Features shape: {sample.x.shape}")  # (nodes, 34 features, 25 timesteps)
print(f"Volatility target: {sample.y.shape}")  # (nodes, 1)
print(f"Risk classes: {sample.risk_class.shape}")  # (nodes,)
```

### Load Standard Dataset

```python
from src.datasets import VNStocksDataset, VNStocksVolatilityDataset

# Standard dataset (price prediction)
dataset = VNStocksDataset(
    root='data/',
    past_window=25,      # 5 weeks
    future_window=1      # 1 day ahead
)

# Volatility dataset
vol_dataset = VNStocksVolatilityDataset(
    root='data/',
    past_window=25,
    future_window=5,          # Predict 5-day volatility
    volatility_window=20      # 20-day rolling window
)

print(f"Dataset size: {len(vol_dataset)}")
sample = vol_dataset[0]
print(f"Sample shape: {sample.x.shape}")  # (nodes, features, timesteps)
print(f"Target shape: {sample.y.shape}")  # (nodes, 1)
```

### Train Hybrid GNN-LSTM Model (NEW)

```python
from src.models import MultiTaskHybridGNN
from src.utils import train
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim

# Create data loaders
train_loader = DataLoader(dataset[:600], batch_size=16, shuffle=True)
test_loader = DataLoader(dataset[600:], batch_size=16, shuffle=False)

# Create hybrid model
model = MultiTaskHybridGNN(
    in_features=34,              # Technical + Macro + Fundamentals
    hidden_size=64,
    num_risk_classes=3,
    num_temporal_layers=2,
    num_graph_layers=2,
    rnn_type='lstm',
    gnn_type='gat',              # Graph Attention Network
    fusion_method='attention',    # Attention-based fusion
    dropout=0.2
)

# Multi-task loss function
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, pred, batch):
        vol_loss = self.mse(pred['volatility'], batch.y)
        risk_loss = self.ce(pred['risk_logits'], batch.risk_class)
        return vol_loss + 0.5 * risk_loss

# Train
history = train(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    criterion=MultiTaskLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=50,
    task_title="hybrid_volatility_risk"
)
```

### Train LSTM Model

```python
from src.models import LSTMModel
from src.utils import train
import torch.optim as optim
import torch.nn as nn

# Create model
model = LSTMModel(in_features=8, hidden_size=64, num_layers=2)

# Train
history = train(
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=0.001),
    criterion=nn.MSELoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    num_epochs=50,
    task_title="volatility_prediction"
)
```

### Evaluate Models

```python
from src.utils import evaluate_model, calculate_metrics, print_metrics_table

# Evaluate LSTM
y_true, y_pred = evaluate_model(model, test_loader)
metrics = calculate_metrics(y_true, y_pred)

# Compare multiple models
results = {
    'ARIMA': arima_metrics,
    'Random Forest': rf_metrics,
    'LSTM': lstm_metrics,
    'GRU': gru_metrics
}

print_metrics_table(results)
```

## Dependencies

Core dependencies:
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical operations
- `torch>=2.0.0` - Deep learning framework
- `torch-geometric>=2.3.0` - Graph neural networks
- `scikit-learn>=1.2.0` - Machine learning models
- `statsmodels>=0.14.0` - ARIMA model
- `matplotlib>=3.6.0` - Visualization
- `seaborn>=0.12.0` - Statistical visualization
- `tqdm>=4.65.0` - Progress bars
- `tensorboard>=2.13.0` - Training monitoring

See [requirements.txt](requirements.txt) for full list.

## Results

Results from model comparison will be saved to:
- `data/analysis/model_comparison_results.json` - Metrics for all models
- `data/analysis/model_comparison.png` - Bar chart comparison
- `data/analysis/predictions_comparison.png` - Prediction scatter plots
- `data/analysis/training_history.png` - Training curves
- `data/analysis/experiment_summary.json` - Full experiment summary

## Research Context

This project is part of NCKH (Nghiên Cứu Khoa Học) research on:
- **Vietnamese FDI stock market analysis**
- **Volatility prediction** for risk assessment
- **Comparison of classical vs modern methods**
- **Graph-based stock relationships**

### Key Contributions
1. **Dataset**: Curated FDI stock data with technical indicators
2. **Graph Structure**: Correlation-based adjacency matrix
3. **Model Comparison**: Systematic evaluation of 4 algorithms
4. **Reproducibility**: Complete pipeline from data to results

## References

Based on the methodology from:
- [SP100AnalysisWithGNNs](https://github.com/timothewt/SP100AnalysisWithGNNs) by timothewt
- Adapted for Vietnamese FDI stocks and volatility prediction

## License

This project is for academic research purposes (NCKH).

## Contact

For questions about this research project, please refer to the NCKH documentation.

---

**Last Updated**: January 2026
