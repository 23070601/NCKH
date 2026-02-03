# 🎯 Predicting the Volatility and Risk Level of Stock Prices of FDI Enterprises Listed in Vietnam

> **Objective**: Predict the average volatility and risk classification over the next 5 days for 98 FDI stocks listed on the Vietnamese stock market

---

## 🚀 CHẠY DỰ ÁN (1 LỆNH)

```bash
./run_pipeline.sh           # Cached (~8s) hoặc lần đầu (~400s)
./run_pipeline.sh --force   # Force chạy lại toàn bộ
```

**Pipeline gồm 8 bước**:
1. Thu thập OHLCV + VNIndex
2. Export features ra CSV
3. Build PyTorch tensors
4. Train models (RandomForest, GradientBoosting, Ridge)
5. Generate base predictions
6. Generate improved predictions (với lag features)
7. Evaluate models
8. Export tables

---

## 📂 CẤU TRÚC DỰ ÁN

```
NCKH/
├── pipeline/             → 8 bước pipeline (CORE)
├── data/
│   ├── raw/             → values.csv (OHLCV), adj.npy
│   ├── features/        → all_features_raw.csv, all_features_processed.csv
│   ├── processed/       → timestep_*.pt (723 timesteps)
│   └── results/         → models, predictions, evaluation
├── src/
│   ├── datasets/        → VNStocksDataset
│   ├── models/          → random_forest.py
│   ├── utils/           → backtest, inference
│   ├── VNStocks.py      → Feature engineering
│   └── data_utils.py    → Data utilities
├── PROJECT_MAP.md       → 🗺️ BẢN ĐỒ CHI TIẾT (XEM ĐÂY)
└── run_pipeline.sh      → Master script
```

👉 **Xem [PROJECT_MAP.md](PROJECT_MAP.md) để hiểu CHI TIẾT từng file làm gì**

## 📊 DỮ LIỆU (INPUT/OUTPUT)

### INPUT (X) - 24 features
**File CSV**: [data/features/all_features_raw.csv](data/features/all_features_raw.csv)

**Features gồm**:
- OHLCV: Open, High, Low, Close, Volume
- Technical: RSI, MACD, MA_5, MA_10, MA_20, BB_UPPER/MID/LOWER, VOL_20
- Returns: DailyLogReturn, ALR1W, ALR2W, ALR1M, ALR2M
- Market: VNIndex_Close, VNIndex_Return

👉 Chi tiết: [DATA_INPUT.md](DATA_INPUT.md)

### OUTPUT (y) - Volatility
**File CSV**: [data/results/exports/all_volatility_labels.csv](data/results/exports/all_volatility_labels.csv)

**Định nghĩa**: Volatility = độ lệch chuẩn của returns trong 20 ngày

**Target**: Dự báo volatility trung bình **5 ngày tới**

👉 Chi tiết: [DATA_OUTPUT.md](DATA_OUTPUT.md)

## Report guide (mở file nào khi thầy hỏi)

### Câu hỏi 1: “Input là gì? data gốc ở đâu?”
- Mở: [DATA_INPUT.md](DATA_INPUT.md)
- File dữ liệu: [data/features/all_features_raw.csv](data/features/all_features_raw.csv)

### Câu hỏi 2: “Output là gì? label tính thế nào?”
- Mở: [DATA_OUTPUT.md](DATA_OUTPUT.md)
- File output: [data/results/exports/all_volatility_labels.csv](data/results/exports/all_volatility_labels.csv)

### Câu hỏi 3: “Feature tính ở đâu?”
- Mở code: [src/VNStocks.py](src/VNStocks.py)

### Câu hỏi 4: “Dataset .pt tạo ở đâu?”
- Mở code: [src/datasets/VNStocksDataset.py](src/datasets/VNStocksDataset.py)

### Câu hỏi 5: “Model train ở đâu?”
- Mở script: [pipeline/04_train_models.py](pipeline/04_train_models.py)

## Features

✅ **Smart Caching**: Reuses previously computed results  
✅ **Force Mode**: `--force` flag triggers full recomputation  
✅ **Automatic Cleanup**: Removes old files before creating new ones  
✅ **Fast Re-runs**: Cached pipeline runs in ~8 seconds  
✅ **Production Ready**: All models saved for deployment

## Project Structure

```
NCKH/
├── run_pipeline.sh                 # Master pipeline runner (recommended)
├── pipeline/                        # Refactored pipeline steps
│   ├── 01_collect_values.py
│   ├── 02b_export_full_features.py
│   ├── 03_build_tensors.py
│   ├── 04_train_models.py
│   ├── 05_base_predictions.py
│   ├── 06_generate_predictions.py
│   ├── 07_evaluate.py
│   ├── 08_export_tables.py
│   └── 09_risk_portfolio.py
├── requirements.txt
├── README.md
├── FINAL_REPORT.md                 # Comprehensive documentation
│
├── data/
│   ├── raw/                        # Raw stock data
│   ├── processed/                  # PyTorch datasets (747 timestep files)
│   ├── features/                   # Full feature tables (all_features_*.csv)
│   └── results/
│       ├── models/                 # Trained models + selectors
│       ├── predictions/            # predictions_*.csv
│       ├── evaluation/             # metrics + plots
│       ├── backtest/               # backtest summaries
│       └── exports/                # CSV exports for input/output
│
└── src/
    ├── VNStocks.py
    ├── data_utils.py
    ├── datasets/
    ├── models/
    └── utils/
```

## Core Scripts

### `run_pipeline.sh` (Recommended)
Runs the full pipeline with caching.

**Usage**:
```bash
./run_pipeline.sh
./run_pipeline.sh --force
```

### Pipeline steps
1. **pipeline/01_collect_values.py** – Collect OHLCV + VNIndex → values.csv + adj.npy
2. **pipeline/02b_export_full_features.py** – Export all features → all_features_raw/processed.csv
3. **pipeline/03_build_tensors.py** – Build timestep_*.pt for training
4. **pipeline/04_train_models.py** – Train RF/GB/Ridge/XGBoost + LSTM + classifier
5. **pipeline/05_base_predictions.py** – Base predictions
6. **pipeline/06_generate_predictions.py** – Improved predictions (lag features)
7. **pipeline/07_evaluate.py** – Metrics + plots
8. **pipeline/08_export_tables.py** – Export CSV tables
9. **pipeline/09_risk_portfolio.py** – VaR/CVaR + CVaR portfolio optimization

## Performance & Speed

| Scenario | Time | Details |
|----------|------|---------|
| **Cached Run** | ~8s | All steps use cached results |
| **Full Rerun** | ~150s | Recomputes all steps with --force |
| **Training Only** | ~20s | Model training only |
| **Evaluation Only** | ~10s | Metrics computation only |

### Regression Performance
```
Baseline Model:    R² = -0.015
Improved Ensemble: R² = 0.2416
Improvement:       +2451%
```

### Classification Performance  
```
Baseline:  33.3% accuracy
Improved:  86.42% accuracy
Improvement: +53.1%
```

### Backtest Results (Top-20 Strategy)
```
Sharpe Ratio:      3.467
Return:            +11.43%
Max Drawdown:      -2.79%
```

## Key Features

- **Lag Engineering**: Captures temporal dependencies (t-1, t-2, t-3)
- **Feature Selection**: SelectKBest with f_regression (203 → 15 features)
- **Ensemble Methods**: Combines Ridge, Random Forest, Gradient Boosting
- **Class Balancing**: SMOTE for imbalanced classification
- **Time-Series Split**: 70% train, 15% val, 15% test

## Hyperparameters

**Random Forest**
- n_estimators: 200
- max_depth: 20
- min_samples_split: 5
- max_features: 'sqrt'

**Gradient Boosting**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 5
- subsample: 0.8

**Ridge Regression**
- alpha: 1.0

**SMOTE**
- k_neighbors: 5

## Dependencies

- scikit-learn (models, feature selection)
- imbalanced-learn (SMOTE)
- pandas, numpy
- torch, torch-geometric
- matplotlib
- yfinance

See requirements.txt for full list.

## Data Structure

**Stock Data**: 98 Vietnamese FDI stocks
**Time Period**: 773 trading days
**Features**: Price, returns, technical indicators (RSI, MACD)
**Graph**: 98×98 correlation matrix

## Workflow

1. Prepare data in data/raw/ and data/processed/
2. Run: `python train_models.py`
3. Run: `python generate_predictions.py`
4. Run: `python evaluate_models.py`
5. Review results in data/results/

## Output Files

- `improved_regressor_*.pkl` - Trained regressor
- `improved_classifier_*.pkl` - Trained classifier
- `feature_selector_*.pkl` - Feature selector
- `predictions_improved_lag_*.csv` - Predictions
- `metrics.json` - Evaluation metrics
- `confusion_matrix.png` - Classification visualization
- `calibration.png` - Prediction calibration

## Notes

- All models use random_state=42 for reproducibility
- Models expect preprocessed torch files in data/processed/
- Feature selection adapts to input data dimensionality
- Time-based train/test split preserves temporal order
- Classification uses volatility percentiles (33.33%, 66.67%)
