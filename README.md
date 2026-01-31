# Vietnamese FDI Stock Volatility Prediction

## Quick Start

**Fastest way - use cached pipeline:**

```bash
./run_pipeline.sh                   # ~8s (cached) or ~150s (first run)
./run_pipeline.sh --force           # Force recomputation of all steps
```

**Or run individual steps:**

```bash
python train_models.py              # Train models with lag features
python create_base_predictions.py   # Create base predictions from models
python generate_predictions.py      # Generate improved predictions
python evaluate_models.py           # Evaluate model performance
```

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
├── train_models.py                 # Model training (with caching)
├── create_base_predictions.py      # Base prediction generation (with caching)
├── generate_predictions.py         # Improved prediction generation (with caching)
├── evaluate_models.py              # Model evaluation (with caching)
├── download_data.py                # Data download utility
├── requirements.txt
├── README.md
├── FINAL_REPORT.md                 # Comprehensive documentation
│
├── data/
│   ├── raw/                        # Raw stock data
│   ├── processed/                  # PyTorch datasets (747 timestep files)
│   ├── features/                   # Feature matrices
│   └── analysis/
│       ├── quick_improvement/      # Trained models (latest only)
│       ├── backtest_improved_lag/  # Backtest results
│       └── evaluation_improved_lag/# Evaluation metrics & visualizations
│
├── notebooks/
│   ├── 0_data_collection.ipynb
│   ├── 1_data_preparation.ipynb
│   ├── 2_data_preparation.ipynb
│   └── 3_model_comparison.ipynb
│
└── src/
    ├── VNStocks.py
    ├── data_utils.py
    ├── macro_data.py
    ├── risk_metrics.py
    ├── datasets/
    ├── models/
    └── utils/
```

## Core Scripts

### `run_pipeline.sh` (Recommended)
Master pipeline runner that orchestrates all steps with smart caching.

**Usage**:
```bash
./run_pipeline.sh                   # Uses cache (fast - ~8s)
./run_pipeline.sh --force           # Full recomputation (~150s)
```

**Caching Logic**:
- Detects existing models, predictions, and evaluations
- Only recomputes if `--force` flag is used or files are missing
- Shows which files are cached and how to force recomputation

### `train_models.py`
Trains regression and classification models with lag features.

**Usage**:
```bash
python train_models.py              # Uses cache if models exist
python train_models.py --force      # Force retraining
```

**Features**: Temporal lags (t-1, t-2, t-3) + rolling statistics  
**Models**: Ridge, Random Forest, Gradient Boosting, Ensemble  
**Classification**: 3-class risk (Low/Medium/High) with SMOTE balancing  

**Results**:
- Regression R²: 0.2416 (vs -0.015 baseline, +2451%)
- Classification Accuracy: 79.56% (vs 33.3% baseline, +46.3%)

### `create_base_predictions.py`
Generates base predictions from trained models on test data.

**Usage**:
```bash
python create_base_predictions.py              # Uses cache if exists
python create_base_predictions.py --force      # Force regeneration
```

**Output**: predictions_*.csv

### `generate_predictions.py`
Applies lag-based improvement to base predictions.

**Usage**:
```bash
python generate_predictions.py              # Uses cache if exists
python generate_predictions.py --force      # Force regeneration
```

**Features**: Lag-aware feature extraction + ensemble regressor  
**Output**: predictions_improved_lag_*.csv

### `evaluate_models.py`
Comprehensive evaluation with metrics and visualizations.

**Usage**:
```bash
python evaluate_models.py              # Uses cache if exists
python evaluate_models.py --force      # Force re-evaluation
```

**Output**: 
- metrics.json (R², MAE, accuracy, confusion matrix)
- confusion_matrix.png
- calibration.png
- classification_report.txt

### `download_data.py`
Downloads stock data from Yahoo Finance.

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
5. Review results in data/analysis/

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
