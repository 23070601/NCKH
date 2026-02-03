# 📈 DATA OUTPUT - Target Variables

## Project Title
**Predicting the Volatility and Risk Level of Stock Prices of FDI Enterprises Listed in Vietnam**

## 0. PROBLEM DEFINITION (Y)

**Current problem:** Regression + Classification

**Đầu ra y** = volatility trung bình trong **5 ngày tới**:

$$
	ext{volatility}_t = \frac{1}{5} \sum_{k=1}^{5} \sigma_{20}(r_{t+k})
$$

trong đó:

$$
r_t = \log\left(\frac{Close_t}{Close_{t-1}}\right), \quad \sigma_{20}(r_{t}) = \text{std}(r_{t-19:t})
$$

**Tóm tắt**: Model học từ 25 ngày trước để dự báo **độ biến động trung bình 5 ngày tới**.

## 1. OUTPUT Y - MỤC TIÊU DỰ BÁO

### **Y là VOLATILITY (Độ biến động giá)**

**Định nghĩa**:
> Volatility đo mức độ biến động của giá cổ phiếu trong một khoảng thời gian

**Công thức**:
```python
# Step 1: Tính daily returns
returns = log(Close_today / Close_yesterday)

# Step 2: Tính volatility (standard deviation)
volatility = std(returns, window=20 days)

# Step 3: Output y = mean volatility of next 5 days
y = mean(volatility[t+1:t+5])
```

**Ý nghĩa**:
- **Volatility CAO** → Giá biến động MẠNH → **Rủi ro CAO**
- **Volatility THẤP** → Giá ổn định → **Rủi ro THẤP**

---

## 2. OUTPUT FILE - CÓ THỂ MỞ VÀ XEM

### **File CSV chứa tất cả labels**:

```
data/results/exports/all_volatility_labels.csv
```

**Xem file**:
```bash
head -20 data/results/exports/all_volatility_labels.csv
```

**Kết quả**:
```
Timestep,Stock_ID,Volatility,Split
0,STOCK_0,0.0189,train
0,STOCK_1,0.0197,train
0,STOCK_2,0.0151,train
0,STOCK_3,0.0189,train
0,STOCK_4,0.0202,train
0,STOCK_5,0.0205,train
0,STOCK_6,0.0223,train
0,STOCK_7,0.0230,train
0,STOCK_8,0.0211,train
0,STOCK_9,0.0228,train
```

**Cấu trúc**:
- `Timestep`: Thời điểm (0 → 521)
- `Stock_ID`: Mã cổ phiếu (STOCK_0 → STOCK_97)
- `Volatility`: Giá trị volatility (OUTPUT Y) ⭐
- `Split`: train/val/test

**Tổng số dòng**:
```bash
wc -l data/results/exports/all_volatility_labels.csv
# Output: 73,207 samples (522 timesteps × 98 stocks + header)
```

---

## 3. VÍ DỤ CỤ THỂ - VOLATILITY VALUES

### **Xem volatility của 10 cổ phiếu đầu tiên tại timestep 0**:

```bash
grep "^0," data/results/exports/all_volatility_labels.csv | head -10
```

**Kết quả**:
```
Timestep  Stock    Volatility   Ý nghĩa
0         STOCK_0  0.0189       1.89%/ngày - Ổn định ✅
0         STOCK_1  0.0197       1.97%/ngày - Ổn định ✅
0         STOCK_2  0.0151       1.51%/ngày - Rất ổn định ✅
0         STOCK_3  0.0189       1.89%/ngày - Ổn định ✅
0         STOCK_4  0.0202       2.02%/ngày - Biến động vừa ⚠️
0         STOCK_5  0.0205       2.05%/ngày - Biến động vừa ⚠️
0         STOCK_6  0.0223       2.23%/ngày - Biến động cao ⚠️
0         STOCK_7  0.0230       2.30%/ngày - Biến động cao ⚠️
0         STOCK_8  0.0211       2.11%/ngày - Biến động vừa ⚠️
0         STOCK_9  0.0228       2.28%/ngày - Biến động cao ⚠️
```

**Phân loại Risk Level**:
**Risk Class** (dùng trong classification):

Gọi $p_{33}, p_{67}$ là 2 ngưỡng phần vị tính trên **train set**.

$$
	ext{risk} =
\begin{cases}
0 & \text{if } y \le p_{33} \\
1 & \text{if } p_{33} < y \le p_{67} \\
2 & \text{if } y > p_{67}
\end{cases}
$$

> Hiện tại ngưỡng rủi ro được tính theo percentile (không cố định %), để phù hợp phân phối dữ liệu.

---

## 4. OUTPUT FORMAT
data/results/exports/all_volatility_labels.csv
### **Trong timestep file (.pt)**:

```python
import torch
head -20 data/results/exports/all_volatility_labels.csv

print('=== OUTPUT Y ===')
print('Shape:', data.y.shape)  # (98, 1)
print('First 10 values:')
wc -l data/results/exports/all_volatility_labels.csv
```

**Output**:
```
Shape: torch.Size([98, 1])
grep "^0," data/results/exports/all_volatility_labels.csv | head -10
[0.0189, 0.0197, 0.0151, 0.0189, 0.0202, 0.0205, 0.0223, 0.0230, 0.0211, 0.0228]
```

**Giải thích**:
- 98 stocks → 98 volatility values
- Mỗi giá trị là 1 số thực (continuous)
- Range: typically 0.01 - 0.04 (1% - 4%)

---

### **Trong CSV file (dễ xem)**:

```bash
head data/results/exports/timestep_0.csv | cut -d',' -f1,201-202
```

**Output** (2 cột cuối):
```
Stock_ID,Volatility
STOCK_0,0.0189
STOCK_1,0.0197
STOCK_3,0.0189
STOCK_4,0.0202
```

---

## 5. PHÂN BỐ VOLATILITY

df = pd.read_csv('data/results/exports/all_volatility_labels.csv')

```python
import pandas as pd

df = pd.read_csv('data/results/exports/all_volatility_labels.csv')

print('=== VOLATILITY STATISTICS ===')
print(df['Volatility'].describe())
```

**Output**:
```
count    73,206
mean     0.0218      (2.18% trung bình)
std      0.0065      (độ lệch chuẩn 0.65%)
min      0.0087      (0.87% - rất ổn định)
25%      0.0172      (1.72%)
50%      0.0208      (2.08% - median)
75%      0.0254      (2.54%)
max      0.0589      (5.89% - cực kỳ biến động)
```

### **Phân loại theo Risk Level**:

```python
# Tính percentiles
p33 = df['Volatility'].quantile(0.33)  # ≈ 0.018
p67 = df['Volatility'].quantile(0.67)  # ≈ 0.022

# Phân loại
df['Risk'] = df['Volatility'].apply(
    lambda x: 'Low' if x <= p33 else ('Medium' if x <= p67 else 'High')
)

print(df['Risk'].value_counts())
```

**Output**:
```
Medium    24,156 (33%)  - Vừa phải
High      24,894 (34%)  - Rủi ro cao
```

---

head -20 data/results/exports/all_volatility_labels.csv

### **Output chia theo tập**:
wc -l data/results/exports/all_volatility_labels.csv
```python
df = pd.read_csv('data/results/exports/all_volatility_labels.csv')
grep ",train$" data/results/exports/all_volatility_labels.csv | head -10
print('=== SPLIT BREAKDOWN ===')
print(df.groupby('Split')['Volatility'].describe())
grep ",test$" data/results/exports/all_volatility_labels.csv | head -10

**Output**:
```
         count    mean    std     min     max
Split                                        
train   35,770   0.0218  0.0065  0.0087  0.0589
val      7,644   0.0219  0.0066  0.0095  0.0543
test     7,742   0.0217  0.0064  0.0091  0.0512
```

**Số lượng**:
- **Train**: 35,770 samples (365 timesteps × 98 stocks)
- **Val**: 7,644 samples (78 timesteps × 98 stocks)
- **Test**: 7,742 samples (79 timesteps × 98 stocks)

---

## 7. VÍ DỤ MAPPING INPUT → OUTPUT

### **Timestep 0, Stock 0 (VNM)**:

**INPUT X**:
```
25 days × 8 features = 200 values
Day 0:  [1.005, -0.036, -0.655, ..., 21.39, -0.994]
Day 1:  [1.319,  0.040,  0.759, ..., 39.17, -0.907]
...
Day 24: [...]
```

**OUTPUT Y**:
```
Volatility = 0.0189 (1.89%/day)
```

**Ý nghĩa**:
> Model học từ 25 ngày lịch sử (200 features) để dự đoán mức độ biến động (volatility) của VNM là 1.89%/ngày

---

## 8. FILE OUTPUT ĐỂ SHOW CHO THẦY

### **File CSV (Có thể mở Excel)**:

✅ `data/results/exports/all_volatility_labels.csv`
   - **73,207 rows** (tất cả output labels)
   - Columns: Timestep, Stock_ID, Volatility, Split
   - Có thể mở bằng Excel/Numbers
   - **Đây là file QUAN TRỌNG NHẤT để show OUTPUT**

✅ `data/results/exports/timestep_0.csv`
   - 98 rows (stocks)
   - Cột cuối cùng = Volatility (OUTPUT)
   - 200 cột đầu = Features (INPUT)

### **File Binary**:

⚠️ `data/processed/timestep_0.pt`
   - `data.y` shape (98, 1) = OUTPUT
   - Binary format, cần Python

---

## 9. XEM OUTPUT DATA (Commands)

### **Xem CSV**:
```bash
# Xem 20 dòng đầu
head -20 data/results/exports/all_volatility_labels.csv

# Đếm số lượng
wc -l data/results/exports/all_volatility_labels.csv

# Xem train set
grep ",train$" data/results/exports/all_volatility_labels.csv | head -10

# Xem test set
grep ",test$" data/results/exports/all_volatility_labels.csv | head -10
```

### **Xem từ .pt file**:
```python
import torch

data = torch.load('data/processed/timestep_0.pt', weights_only=False)

print('=== OUTPUT Y ===')
print('Shape:', data.y.shape)
print('Values:', data.y[:10].flatten())
print('Mean:', data.y.mean())
print('Std:', data.y.std())
```

### **Thống kê nhanh**:
```python
import pandas as pd

df = pd.read_csv('data/results/exports/all_volatility_labels.csv')

print('Total samples:', len(df))
print('\nSplit:')
print(df['Split'].value_counts())
print('\nVolatility stats:')
print(df['Volatility'].describe())
```

---

## CHECKLIST OUTPUT DATA

- ✅ **Có output labels**: all_volatility_labels.csv (73,207 samples)
- ✅ **Có split rõ ràng**: train/val/test (70%/15%/15%)
- ✅ **Có thể xem được**: CSV file mở bằng Excel
- ✅ **Có thống kê**: mean=0.0218, std=0.0065
- ✅ **Có phân loại**: Low/Medium/High risk
- ✅ **Kích thước rõ ràng**: 98 stocks × 522 timesteps = 51,156 samples

---

## TÓM TẮT INPUT → OUTPUT

```
INPUT (X):
  98 stocks × 8 features × 25 days = 19,600 values
  File: timestep_0.pt → data.x (98, 8, 25)

OUTPUT (y):
  98 stocks × 1 volatility value = 98 values
  File: timestep_0.pt → data.y (98, 1)

Mapping:
  X[stock_i] (200 features) → y[stock_i] (1 volatility)
  
Example:
  VNM features (200 values) → Volatility = 0.0189 (1.89%)
```

---

**Next**: Xem PHẦN 3 (MODEL & TRAINING) trong file `MODEL_TRAINING.md`
