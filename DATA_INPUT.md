# 📊 DATA INPUT - Input Features

## Project Title
**Predicting the Volatility and Risk Level of Stock Prices of FDI Enterprises Listed in Vietnam**

## 0. PROBLEM STATEMENT

**Objective**: Predict volatility (price fluctuation) **average over the next 5 days** and classify risk level for each FDI stock.

**Horizon**:
- $t$ = ngày hiện tại
- **Output**: volatility trung bình của **$t+1$ → $t+5$**

**Lưu ý**: Bài toán hiện tại là **regression** (dự báo giá trị volatility).

## 1. RAW DATA (Dữ liệu gốc từ thị trường)

### **File CSV có thể mở và xem**:

```
📂 data/features/
├── tickers.csv               (399 B)   - Danh sách 98 mã cổ phiếu
├── close_matrix.csv          (1.3 MB)  - Giá đóng cửa
├── dailylogreturn_matrix.csv (1.4 MB)  - Tỷ suất sinh lời
├── rsi_matrix.csv            (1.3 MB)  - Chỉ số RSI
└── macd_matrix.csv           (1.4 MB)  - Chỉ số MACD
```

### **Những input hiện có (đã có)**

**Nhóm Giá & kỹ thuật (OHLCV + indicators)**:
- Open, High, Low, Close, Volume
- DailyLogReturn
- RSI (14), MACD (12,26)
- MA_5, MA_10, MA_20
- Bollinger Bands: BB_UPPER, BB_MID, BB_LOWER
- VOL_20 (rolling volatility)
- ALR1W, ALR2W, ALR1M, ALR2M

**Nhóm Ngoại sinh**:
- VNIndex_Close, VNIndex_Return

### **Những input chưa có (thiếu)**

**Chưa có**:
- Lãi suất, tin tức, sentiment, chỉ số ngành

### **Xem danh sách cổ phiếu**:

```bash
cat data/features/tickers.csv
```

**Kết quả** (98 mã):
```
ticker
VNM
SAB
MSN
VIC
VHM
HPG
VCB
BID
CTG
GAS
... (98 stocks total)
```

---

### **Xem giá đóng cửa (close_matrix.csv)**:

```bash
head -5 data/features/close_matrix.csv | cut -d',' -f1-6
```

**Kết quả**:
```
Date        AAA      ACB      AGG      ANV      ASM
2022-01-03  119.08   138.30   63.26    80.54    110.68
2022-01-04  120.33   135.77   62.86    80.52    111.00
2022-01-05  123.68   137.73   65.00    82.11    110.00
2022-01-06  126.16   137.49   66.32    80.82    109.62
```

**Cấu trúc**:
- **Rows**: 773 ngày giao dịch (2022-01-03 → 2024-12-31)
- **Columns**: 99 cột (1 Date + 98 stocks)
- **Values**: Giá đóng cửa (đơn vị: nghìn VNĐ)

---

### **Xem daily log return (dailylogreturn_matrix.csv)**:

```bash
head -5 data/features/dailylogreturn_matrix.csv | cut -d',' -f1-6
```

**Kết quả**:
```
Date        AAA          ACB           AGG           ANV
2022-01-03  0.0105       -0.0184       -0.0065       -0.0002
2022-01-04  0.0105       -0.0184       -0.0065       -0.0002
2022-01-05  0.0277        0.0144        0.0335        0.0197
2022-01-06  0.0199       -0.0018        0.0203       -0.0157
```

**Công thức**:
```python
DailyLogReturn = log(Close_today / Close_yesterday)
```

**Ý nghĩa**:
- Dương (+): Giá tăng → Sinh lời
- Âm (-): Giá giảm → Lỗ
- Example: 0.0105 = tăng 1.05%

---

## 2. INPUT X - ĐẶC TRƯNG ĐẦU VÀO

### **8 Features cho mỗi cổ phiếu**:

| # | Feature Name | File Source | Ý nghĩa |
|---|--------------|-------------|---------|
| 1 | **Close** | close_matrix.csv | Giá đóng cửa |
| 2 | **DailyLogReturn** | dailylogreturn_matrix.csv | Tỷ suất sinh lời ngày |
| 3 | **RSI** | rsi_matrix.csv | Chỉ số sức mạnh (0-100) |
| 4 | **MACD** | macd_matrix.csv | Chỉ số xu hướng |
| 5 | **ALR1W** | Tính từ Close | Tỷ suất sinh lời 1 tuần |
| 6 | **ALR2W** | Tính từ Close | Tỷ suất sinh lời 2 tuần |
| 7 | **ALR1M** | Tính từ Close | Tỷ suất sinh lời 1 tháng |
| 8 | **ALR2M** | Tính từ Close | Tỷ suất sinh lời 2 tháng |

### **Temporal Window (Cửa sổ thời gian)**:

```
Input sử dụng 25 ngày lịch sử:

Day 0 (oldest)  → [Feature1, Feature2, ..., Feature8]
Day 1           → [Feature1, Feature2, ..., Feature8]
...
Day 24 (newest) → [Feature1, Feature2, ..., Feature8]

Total: 8 features × 25 days = 200 values per stock
```

### **Kích thước Input X cho 1 timestep**:

```
98 stocks × 8 features × 25 days = 19,600 values
```

**File format**: `data/processed/timestep_0.pt`
```python
import torch
data = torch.load('data/processed/timestep_0.pt')
print(data.x.shape)  # torch.Size([98, 8, 25])
```

---

## 3. VÍ DỤ CỤ THỂ - STOCK ĐẦU TIÊN (VNM)

### **Input X cho VNM tại timestep 0**:

```python
import torch
data = torch.load('data/processed/timestep_0.pt')

vnm_features = data.x[0]  # Stock index 0 = VNM
print('VNM Input Shape:', vnm_features.shape)  # (8, 25)

# First 5 days, all 8 features
print('First 5 days:')
print(vnm_features[:, :5].T)
```

**Output**:
```
         Feature0  Feature1  Feature2  Feature3  Feature4  Feature5  Feature6  Feature7
Day 0:   1.005     -0.036    -0.655    -1.795    -0.366    -1.035     21.39    -0.994
Day 1:   1.319      0.040     0.759    -0.316    -0.366    -1.035     39.17    -0.907
Day 2:   1.292     -0.003    -0.041    -0.161    -0.200    -1.035     38.56    -0.850
Day 3:   1.161     -0.017    -1.623    -0.789    -0.402    -1.035     36.06    -0.905
Day 4:   1.141     -0.003    -0.928    -0.810    -0.772    -1.035     38.17    -0.954
```

**Giải thích**:
- Day 0 = Ngày xa nhất (25 ngày trước)
- Day 24 = Ngày gần nhất (hôm nay)
- Model sẽ học từ 25 ngày này để dự đoán volatility tương lai

---

## 4. TỔNG HỢP INPUT

### **Số liệu**:

```
Raw Data:
  ├─ Period: 2022-01-03 → 2024-12-31
  ├─ Trading days: 773 days
  ├─ Stocks: 98 FDI stocks
  └─ Total data points: 773 × 98 = 75,754

Temporal Snapshots:
  ├─ Window size: 25 days
  ├─ Total timesteps: 773 - 25 = 748 (actually 522 after processing)
  └─ Files: timestep_0.pt → timestep_521.pt

Input per timestep:
  ├─ Stocks: 98
  ├─ Features: 8
  ├─ Days: 25
  └─ Total values: 98 × 8 × 25 = 19,600
```

---

## 5. XEM INPUT DATA (Commands để check)

### **Xem CSV**:
```bash
# Danh sách cổ phiếu
head data/features/tickers.csv

# Giá đóng cửa 5 ngày đầu, 5 cổ phiếu đầu
head -5 data/features/close_matrix.csv | cut -d',' -f1-6

# Returns 5 ngày đầu
head -5 data/features/dailylogreturn_matrix.csv | cut -d',' -f1-6

# RSI 5 ngày đầu
head -5 data/features/rsi_matrix.csv | cut -d',' -f1-6
```

### **Xem Processed Tensor**:
```python
import torch

# Load 1 timestep
data = torch.load('data/processed/timestep_0.pt', weights_only=False)

print('=== INPUT X ===')
print('Shape:', data.x.shape)  # (98, 8, 25)
print('First stock, first 3 days:')
print(data.x[0, :, :3])
```

### **Export to CSV để dễ xem**:
```bash
# Đã tạo sẵn
ls -lh data/results/exports/
# → timestep_0.csv, timestep_1.csv, etc.
```

---

## 6. FILE INPUT ĐỂ SHOW CHO THẦY

### **File CSV (Có thể mở Excel/Numbers)**:

✅ `data/features/tickers.csv`
   - Mở được bằng Excel
   - 98 dòng = 98 mã cổ phiếu

✅ `data/features/close_matrix.csv`
   - Mở được bằng Excel
   - 773 rows × 99 columns
   - Giá đóng cửa thực tế

✅ `data/features/dailylogreturn_matrix.csv`
   - Mở được bằng Excel
   - Returns đã tính

✅ `data/results/exports/timestep_0.csv`
   - Mở được bằng Excel
   - 98 rows (stocks) × 202 columns (features + volatility)

### **File Binary (Cần Python để đọc)**:

⚠️ `data/processed/timestep_0.pt`
   - Binary format (PyTorch)
   - Cần load bằng `torch.load()`
   - Chứa X, y, graph structure

---

## CHECKLIST INPUT DATA

- ✅ **Có raw data**: 4 CSV files (close, return, rsi, macd)
- ✅ **Có danh sách stocks**: tickers.csv (98 mã)
- ✅ **Có processed data**: 522 timestep files
- ✅ **Có thể xem được**: CSV files mở bằng Excel
- ✅ **Có thể verify**: Python scripts để check
- ✅ **Kích thước rõ ràng**: 98 stocks × 8 features × 25 days

---

**Next**: Xem PHẦN 2 (OUTPUT) trong file `DATA_OUTPUT.md`
