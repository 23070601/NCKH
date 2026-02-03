# ✅ DỌN DẸP HOÀN TẤT

**Project**: Predicting the Volatility and Risk Level of Stock Prices of FDI Enterprises Listed in Vietnam

## 📊 Trước và Sau

| | Trước | Sau | Giảm |
|---|---|---|---|
| Python files | 29 | 18 | -11 (38%) |
| CSV files | 751 | 12 | -739 (98%) |
| Tổng files | ~780+ | ~30 | **-96%** |

## 🗑️ Đã Xóa

1. **720 timestep CSVs** → Giữ 3 files mẫu (timestep_0, 100, 500)
2. **17 feature matrix CSVs riêng lẻ** → Gộp thành `all_features_raw.csv` và `all_features_processed.csv`
3. **4 CSV files trùng lặp**:
   - `data/processed/values.csv` → TRÙNG với `data/raw/values.csv`
   - `data/processed/values_enriched.csv` → TRÙNG với `all_features_raw.csv`
   - `data/results/exports/full_dataset_raw.csv` → TRÙNG với `all_features_raw.csv`
   - `data/results/exports/full_dataset_processed.csv` → TRÙNG với `all_features_processed.csv`
4. **11 code files không dùng** → Chuyển vào `archive/unused_code/`:
   - lstm.py, gru.py, hybrid_gnn_lstm.py, hybrid_predictor.py, arima.py
   - train.py, evaluate.py, evaluate_predictions.py
   - EnhancedDataset.py, macro_data.py, risk_metrics.py

## 📂 Cấu Trúc Hiện Tại (GỌN)

```
NCKH/
├── pipeline/          → 8 scripts (01-08)
├── data/
│   ├── raw/          → values.csv (31M), adj.npy, fdi_stocks_list.csv
│   ├── features/     → all_features_raw.csv (30M), all_features_processed.csv (32M), tickers.csv
│   ├── processed/    → timestep_*.pt (723 files - PyTorch tensors)
│   └── results/      
│       ├── models/           → trained .pkl files + predictions_*.csv
│       ├── predictions/      → predictions_*.csv, predictions_improved_*.csv  
│       ├── evaluation/       → metrics.json, confusion_matrix.png
│       └── exports/
│           ├── all_volatility_labels.csv (2.3M)
│           └── timesteps/    → 3 sample CSVs (timestep_0, 100, 500)
├── src/
│   ├── datasets/     → VNStocksDataset.py
│   ├── models/       → random_forest.py
│   ├── utils/        → backtest.py, inference.py
│   ├── VNStocks.py   → Feature engineering (QUAN TRỌNG)
│   └── data_utils.py
├── archive/          → notebooks/, unused_code/
├── PROJECT_MAP.md    → 🗺️ BẢN ĐỒ CHI TIẾT
├── README.md
├── DATA_INPUT.md
├── DATA_OUTPUT.md
└── run_pipeline.sh
```

## 📋 12 CSV Files Còn Lại (Mỗi File Có Ý Nghĩa)

### Data nguồn (2 files):
1. `data/raw/values.csv` (31M) - OHLCV data gốc từ vnstock
2. `data/raw/fdi_stocks_list.csv` (6.4K) - Danh sách 98 mã FDI

### Features (3 files):
3. `data/features/all_features_raw.csv` (30M) - **TẤT CẢ 24 features (RAW)**
4. `data/features/all_features_processed.csv` (32M) - **TẤT CẢ 24 features (normalized)**
5. `data/features/tickers.csv` (399B) - Danh sách tickers

### Labels/Output (1 file):
6. `data/results/exports/all_volatility_labels.csv` (2.3M) - TẤT CẢ labels (y)

### Predictions (3 files):
7. `data/results/models/predictions_20260201_111752.csv` (165K) - Per-sample predictions
8. `data/results/predictions/predictions_20260201_111804.csv` (1.4M) - Base predictions
9. `data/results/predictions/predictions_improved_lag_20260201_111813.csv` (1.3M) - Improved

### Samples (3 files):
10. `data/results/exports/timesteps/timestep_0.csv` (940K) - Timestep đầu
11. `data/results/exports/timesteps/timestep_100.csv` (942K) - Timestep giữa
12. `data/results/exports/timesteps/timestep_500.csv` (941K) - Timestep cuối

## 🎯 Files Quan Trọng Nhất

1. **PROJECT_MAP.md** - Giải thích từng file làm gì
2. **all_features_raw.csv** - TẤT CẢ 24 features (75,754 rows)
3. **all_features_processed.csv** - Features đã chuẩn hóa
4. **pipeline/04_train_models.py** - Training logic
5. **src/VNStocks.py** - Feature engineering

## 📝 Cách Dùng

### Xem bản đồ dự án:
```bash
cat PROJECT_MAP.md
```

### Xem features:
```bash
head -5 data/features/all_features_raw.csv
```

### Chạy pipeline:
```bash
./run_pipeline.sh
```

---

**Dự án giờ đã sạch sẽ và dễ hiểu!** 🎉
