# Vietnamese FDI Stock Market Analysis

## Research Project: Predicting Volatility and Risk Level of Stock Prices for FDI Enterprises in Vietnam

This repository contains code and data for analyzing Vietnamese FDI (Foreign Direct Investment) companies using Graph Neural Networks (GNNs).

## Project Structure

```
code/
├── notebooks/
│   └── 1_data_preparation.ipynb    # Data preparation and feature engineering
├── src/
│   ├── VNStocks.py                 # Stock data collection pipeline
│   └── utils.py                    # Utility functions
├── data/
│   ├── fdi_stocks_list.csv        # List of FDI companies to analyze
│   ├── stocks.csv                 # Stock metadata (generated)
│   ├── fundamentals.csv           # Fundamental features (generated)
│   ├── values.csv                 # Daily closing prices (generated)
│   └── adj.npy                    # Adjacency matrix for GNN (generated)
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install torch torch-geometric
pip install vnstock  # Optional: For real Vietnamese stock data
```

### 2. Data Collection Pipeline

The data collection process follows these steps:

#### Step 1: Define FDI Stock List
Edit `data/fdi_stocks_list.csv` with Vietnamese companies that have FDI involvement:
- Ticker symbol
- Company name
- Sector
- Exchange (HOSE, HNX, UPCOM)
- FDI status (High, Medium, Low)
- Market cap category

#### Step 2: Run Data Collection

```bash
cd src
python VNStocks.py
```

This will generate:
- `stocks.csv`: Stock metadata
- `fundamentals.csv`: PE ratio, ROE, debt-to-equity, etc.
- `values.csv`: Aligned daily closing prices for all stocks
- `adj.npy`: Graph adjacency matrix based on correlation

#### Step 3: Data Preparation
Open and run `notebooks/1_data_preparation.ipynb` to:
- Load and preprocess data
- Calculate returns and log returns
- Compute volatility metrics (historical, rolling, EWMA)
- Calculate risk indicators (VaR, CVaR, Sharpe ratio, etc.)
- Engineer features for GNN input
- Create graph structures with PyTorch Geometric

## Data Files Description

### Generated Data Files

1. **stocks.csv**
   - Metadata for each stock (ticker, name, sector, exchange, FDI status)
   
2. **fundamentals.csv**
   - Fundamental metrics: PE ratio, PB ratio, ROE, debt-to-equity, dividend yield, market cap, beta
   
3. **values.csv**
   - Time-series of daily closing prices
   - Columns: Date, Ticker1, Ticker2, ..., TickerN
   - All stocks aligned by trading dates
   
4. **adj.npy**
   - Adjacency matrix (N x N) where N = number of stocks
   - Binary matrix: 1 if correlation > threshold, 0 otherwise
   - Used to construct graph edges for GNN

## Research Methodology

### 1. Data Collection
- Identify Vietnamese companies with significant FDI
- Collect historical price data (2+ years recommended)
- Gather fundamental financial metrics

### 2. Feature Engineering
- **Node Features**: Mean return, volatility, skewness, kurtosis, max/min returns, VaR
- **Edge Features**: Correlation coefficients between stock returns
- **Graph Structure**: Correlation-based adjacency matrix

### 3. Volatility & Risk Metrics
- Historical volatility (annualized standard deviation)
- Rolling volatility (20-day window)
- Exponential weighted moving average (EWMA) volatility
- Value at Risk (VaR 95%)
- Conditional Value at Risk (CVaR)
- Sharpe ratio, Sortino ratio
- Maximum drawdown

### 4. Graph Neural Network
- Nodes represent stocks
- Edges represent correlations above threshold
- GNN learns from graph structure to predict volatility and risk

## Usage Example

```python
from src.VNStocks import VNStocksDataset

# Initialize dataset
dataset = VNStocksDataset(
    stock_list_path='data/fdi_stocks_list.csv',
    start_date='2022-01-01',
    end_date='2024-12-31',
    data_dir='data'
)

# Collect data
dataset.collect_price_data(source='vnstock')
dataset.collect_fundamentals()

# Save processed data
dataset.save_all_data()

# Get summary
summary = dataset.get_summary()
print(summary)
```

## Real Data Sources for Vietnamese Stocks

### Option 1: vnstock Library (Recommended)
```bash
pip install vnstock
```

```python
from vnstock import stock_historical_data

df = stock_historical_data(
    symbol='VNM',
    start_date='2022-01-01',
    end_date='2024-12-31'
)
```

### Option 2: VNDirect API
- Register at: https://www.vndirect.com.vn/
- Use their market data API

### Option 3: Manual Data Collection
- Download from: https://www.cophieu68.vn/
- Or: https://www.vietstock.vn/

## FDI Company Selection Criteria

Companies are selected based on:
1. **High FDI involvement**: Foreign ownership > 30%
2. **Listed on HOSE**: Main exchange for largest companies
3. **Market capitalization**: Large or mid-cap stocks
4. **Sector diversity**: Coverage across multiple sectors
5. **Liquidity**: Average daily trading volume > threshold

## Next Steps

1. ✅ Create stock list for FDI companies
2. ✅ Build data collection utilities
3. ✅ Implement data processing pipeline
4. ⏳ Collect real Vietnamese stock data
5. ⏳ Build GNN model architecture
6. ⏳ Train and evaluate model
7. ⏳ Analyze results and write research paper

## Reference

This project is based on the methodology from:
- [SP100AnalysisWithGNNs](https://github.com/timothewt/SP100AnalysisWithGNNs)
- Adapted for Vietnamese FDI stock market analysis

## License

MIT License - Academic Research Use
