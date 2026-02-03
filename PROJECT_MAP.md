# 🗺️ PROJECT MAP - What Each File Does

**Project Title**: Predicting the Volatility and Risk Level of Stock Prices of FDI Enterprises Listed in Vietnam

> **Purpose**: Clear explanation of each file/folder in the project for easy reference

---

## 📁 CẤU TRÚC TỔNG QUAN

```
NCKH/
├── 🔧 pipeline/          → 9 BƯỚC CHẠY DỰ ÁN (QUAN TRỌNG NHẤT)
├── 📊 data/              → DỮ LIỆU (input, output, kết quả)
├── 🧰 src/               → SOURCE CODE (dataset, models, utils)
├── 📄 *.md               → TÀI LIỆU HƯỚNG DẪN
└── 🏃 run_pipeline.sh    → CHẠY TOÀN BỘ (1 LỆNH)
```

---

## 🔧 PIPELINE (9 BƯỚC - CORE CỦA DỰ ÁN)

**Chạy toàn bộ**: `./run_pipeline.sh`

| File | Mục đích | Input | Output |
|------|----------|-------|--------|
| **01_collect_values.py** | Thu thập OHLCV + VNIndex | vnstock API | `data/raw/values.csv` |
| **02b_export_full_features.py** | Export TẤT CẢ features vào 1 file | values.csv | `all_features_raw.csv`, `all_features_processed.csv` |
| **03_build_tensors.py** | Build PyTorch tensors | values.csv + adj.npy | `data/processed/timestep_*.pt` (723 files) |
| **04_train_models.py** | Train models (RF, GB, Ridge, XGBoost, LSTM) | timestep_*.pt | `data/results/models/*.pkl`, `.pt` |
| **05_base_predictions.py** | Dự đoán baseline | trained models | `predictions_*.csv` |
| **06_generate_predictions.py** | Dự đoán improved (với lag) | base predictions | `predictions_improved_*.csv` |
| **07_evaluate.py** | Đánh giá model | predictions | `metrics.json`, `confusion_matrix.png` |
| **08_export_tables.py** | Export timesteps ra CSV (ĐÃ XÓA) | timestep_*.pt | timestep CSVs (giữ 3 files mẫu) |
| **09_risk_portfolio.py** | VaR/CVaR + CVaR portfolio optimization | all_features_raw.csv | `data/results/portfolio/*.csv` |

---

## 📊 DATA (Dữ liệu)

### `data/raw/` - Dữ liệu GỐC
```
fdi_stocks_list.csv    → Danh sách 98 mã FDI
values.csv             → OHLCV + features (75,754 rows × 24 cols)
adj.npy                → Ma trận kề (adjacency matrix) cho GNN
```

### `data/features/` - Features ĐÃ TÍNH
```
tickers.csv                    → 98 mã cổ phiếu
all_features_raw.csv           → TẤT CẢ 24 features (RAW - chưa chuẩn hóa)
all_features_processed.csv     → TẤT CẢ 24 features (ĐÃ chuẩn hóa)
```
**24 features**: Open, High, Low, Close, Volume, NormClose, DailyLogReturn, ALR1W, ALR2W, ALR1M, ALR2M, RSI, MACD, MA_5, MA_10, MA_20, BB_MID, BB_UPPER, BB_LOWER, VOL_20, VNIndex_Close, VNIndex_Return

### `data/processed/` - Tensors cho training
```
timestep_0.pt ... timestep_722.pt   → 723 timesteps (PyTorch Data objects)
```

### `data/results/` - KẾT QUẢ
```
models/
├── improved_regressor_*.pkl     → Trained regression model
├── improved_classifier_*.pkl    → Trained classification model
├── feature_selector_*.pkl       → SelectKBest selector
├── summary_*.json               → Kết quả training
└── predictions_*.csv            → Per-sample predictions

predictions/
├── predictions_*.csv                   → Base predictions
└── predictions_improved_lag_*.csv      → Improved predictions

evaluation/
├── metrics.json                 → R², RMSE, Accuracy, F1
└── confusion_matrix.png         → Ma trận confusion

backtest/
└── backtest_summary_*.json      → Backtest results

exports/
├── all_volatility_labels.csv   → TẤT CẢ labels (y)
└── timesteps/
    ├── timestep_0.csv          → Mẫu timestep đầu
    ├── timestep_100.csv        → Mẫu timestep giữa
    └── timestep_500.csv        → Mẫu timestep cuối

portfolio/
├── risk_metrics.csv            → VaR/CVaR per stock
└── portfolio_cvar.csv          → CVaR-optimized weights
```

### `data/train_test_split.json` - Chia tập dữ liệu
Timeline split: 70% train, 15% val, 15% test

---

## 🧰 SRC (Source Code)

### `src/datasets/` - Dataset classes
```
VNStocksDataset.py     → Dataset chính (load OHLCV, build tensors)
```

### `src/models/` - Model implementations
**ĐANG DÙNG:**
- `random_forest.py` → RandomForest wrapper (pipeline/04)
- `lstm.py` → LSTM model (pipeline/04)

### `src/utils/` - Utilities
```
train.py                 → Training utilities (CHƯA DÙNG - pipeline tự implement)
evaluate.py              → Evaluation utilities (CHƯA DÙNG)
evaluate_predictions.py  → Evaluate predictions (CHƯA DÙNG)
inference.py             → Inference utilities (DÙNG TRONG pipeline/05, 06)
backtest.py              → Backtesting (DÙNG SAU evaluate)
```

### `src/` - Core modules
```
VNStocks.py      → VNStocks class - FEATURE ENGINEERING (QUAN TRỌNG)
data_utils.py    → Download VNIndex, load data
```

---

## 📄 TÀI LIỆU (Documentation)

| File | Nội dung |
|------|----------|
| **README.md** | Hướng dẫn CHẠY dự án |
| **DATA_INPUT.md** | Giải thích INPUT (X) - 24 features |
| **DATA_OUTPUT.md** | Giải thích OUTPUT (y) - volatility |
| **PROJECT_MAP.md** | File này - BẢN ĐỒ dự án |

---

## 🏃 CHẠY DỰ ÁN

### Cách 1: Chạy toàn bộ (KHUYẾN NGHỊ)
```bash
./run_pipeline.sh           # Cached (~8s)
./run_pipeline.sh --force   # Force recompute (~400s)
```

### Cách 2: Chạy từng bước
```bash
python pipeline/01_collect_values.py
python pipeline/02b_export_full_features.py
python pipeline/03_build_tensors.py
python pipeline/04_train_models.py
python pipeline/05_base_predictions.py
python pipeline/06_generate_predictions.py
python pipeline/07_evaluate.py
python pipeline/09_risk_portfolio.py
```

---

## 🎯 KHI THẦY HỎI - MỞ FILE NÀO?

### "Input là gì?"
→ Mở: `DATA_INPUT.md`  
→ File CSV: `data/features/all_features_raw.csv`

### "Output là gì?"
→ Mở: `DATA_OUTPUT.md`  
→ File CSV: `data/results/exports/all_volatility_labels.csv`

### "Feature tính ở đâu?"
→ Mở code: `src/VNStocks.py` (dòng 50-200)

### "Model nào đang dùng?"
→ Mở: `pipeline/04_train_models.py`  
→ Models: RandomForest, GradientBoosting, Ridge, Ensemble

### "Kết quả thế nào?"
→ Mở: `data/results/evaluation/metrics.json`

### "Dự đoán thế nào?"
→ Mở: `data/results/predictions/predictions_improved_lag_*.csv`

---

## 📊 THỐNG KÊ DỰ ÁN

- **Python files**: 29 files
- **CSV files**: 28 files (đã xóa 720 timestep CSVs)
- **Tổng dữ liệu**: 75,754 rows × 24 features
- **Timesteps**: 723 (train: 506, val: 108, test: 109)
- **Models**: 3 regressors + 1 ensemble + 1 classifier
- **Metrics**: R² = 0.92, Accuracy = 89.7%

---

## 🗑️ ĐÃ XÓA / ARCHIVE

- ❌ `data/results/exports/timesteps/` → Xóa 720 files, giữ 3 mẫu
- ❌ `data/features/*_matrix.csv` → Đã gộp vào `all_features_*.csv`
- ❌ `notebooks/` → Chuyển vào `archive/`
- ❌ Files wrapper cũ: download_data.py, train_models.py, etc.

---

## ⚠️ LƯU Ý

1. **LSTM/GRU/Hybrid models**: Đã implement nhưng CHƯA dùng trong pipeline
2. **Feature matrices riêng lẻ**: ĐÃ XÓA, dùng `all_features_*.csv`
3. **Timestep CSVs**: Chỉ giữ 3 files mẫu (0, 100, 500)
4. **Utils**: Một số utils chưa dùng, có thể xóa sau

---

**Cập nhật**: 2026-02-01
