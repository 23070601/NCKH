# Algorithm Comparison Framework for FDI Stock Volatility Prediction

## 🎯 Multi-Algorithm Testing Strategy

Your research follows a **systematic algorithm comparison approach** rather than using the same GNN-only strategy as the reference project:

| Nhóm     | Thuật toán              | Vai trò           | Ưu điểm |
|----------|-------------------------|-------------------|---------|
| **Baseline** | Historical Mean / ARIMA | Mốc so sánh | Đơn giản, nhanh, dễ giải thích |
| **ML** | Random Forest | Phi tuyến | Capture non-linear patterns, feature importance |
| **DL** | LSTM | Quan hệ thời gian | Long-term dependencies, temporal modeling |
| **DL (opt)** | GRU | So sánh nội bộ | Faster training, similar performance to LSTM |

---

## 📊 Implementation Overview

### Phase 1: Data Collection & Preparation
```
notebooks/0_data_collection.ipynb     ← Generate raw data
  ↓
notebooks/1_data_preparation.ipynb    ← Feature engineering
  ↓ volatility_data, features, train/test splits
```

### Phase 2: Algorithm Comparison ⭐ NEW
```
notebooks/2_model_comparison.ipynb    ← Test all 4 algorithm groups
  ↓
  Baseline:  Historical Mean, ARIMA
  ML:        Random Forest
  DL:        LSTM, GRU
  ↓
  Output: Comparison table, best algorithm recommendation
```

### Phase 3: Optimal Algorithm Implementation (Future)
```
notebooks/3_optimal_model.ipynb       ← Fine-tune best algorithm
  ↓
notebooks/4_fdi_analysis.ipynb        ← FDI-specific insights
  ↓
notebooks/5_visualization.ipynb       ← Results & visualization
```

---

## 🔧 New Files Added

### 1. Model Comparison Framework (`src/model_comparison.py`)
Complete implementation of all algorithm classes:

```python
# Baseline Models
HistoricalMeanModel()        # Simple moving average prediction
ARIMAModel()                 # Time-series ARIMA forecasting

# ML Models
RandomForestVolatilityModel() # Non-linear pattern capture

# DL Models
LSTMVolatilityModel()         # LSTM with dropout regularization
GRUVolatilityModel()          # GRU (faster than LSTM)

# Advanced DL (Optional)
TemporalGCNVolatilityModel()  # GNN-based (for comparison)

# Utilities
ModelComparator()             # Evaluation framework
create_sequences()            # RNN data preparation
split_data()                  # Train/val/test splitting
```

### 2. Algorithm Comparison Notebook (`notebooks/2_model_comparison.ipynb`)
**11 Sections:**
1. ✅ Import libraries
2. ✅ Load & explore data
3. ✅ Preprocessing & feature engineering
4. ✅ Time-series train/test split
5. ✅ Baseline models (Historical Mean, ARIMA)
6. ✅ Random Forest training
7. ✅ LSTM implementation
8. ✅ GRU implementation
9. ✅ Evaluation metrics comparison
10. ✅ Visualize all predictions
11. ✅ **Optimal algorithm recommendation**

---

## 🎓 How This Differs From Reference Project

### Reference Project (S&P 100)
- Focus: Multiple GNN architectures (T-GCN, STGCN, GAT, A3T-GCN)
- Approach: Advanced deep learning only
- Purpose: Demonstrate GNN capabilities

### Your Project (VN FDI)
- Focus: **Systematic algorithm comparison** across all complexity levels
- Approach: Baseline → ML → DL (including GNN option)
- Purpose: **Find optimal algorithm for Vietnamese FDI stocks** + FDI analysis
- Innovation: Tests if complex GNNs are necessary or if simpler models suffice

---

## 📈 Expected Output: Comparison Table

After running `notebooks/2_model_comparison.ipynb`:

```
================================================================================
MODEL COMPARISON TABLE
================================================================================
                    Model    Group       RMSE          MAE         MAPE
0           Random Forest       ML    0.012543    0.008934    1.234%
1                   LSTM        DL    0.013821    0.009456    1.456%
2                    GRU        DL    0.014102    0.009821    1.523%
3            ARIMA(1,1,1)  Baseline    0.015634    0.010234    1.789%
4        Historical Mean  Baseline    0.018945    0.012456    2.134%
================================================================================

Improvement over best baseline: 19.87%

✓ OPTIMAL ALGORITHM: Random Forest (ML)
```

---

## 🚀 How To Use

### Step 1: Run Data Collection & Preparation
```bash
# Open and execute:
notebooks/0_data_collection.ipynb
notebooks/1_data_preparation.ipynb
```

### Step 2: Compare All Algorithms
```bash
# Open and execute:
notebooks/2_model_comparison.ipynb

# This will:
# ✓ Train 5 different algorithms
# ✓ Compare performance metrics
# ✓ Recommend optimal choice
# ✓ Show prediction visualizations
```

### Step 3: Fine-tune Optimal Algorithm
```bash
# You'll create:
notebooks/3_optimal_model.ipynb

# Which will:
# ✓ Hyperparameter tuning
# ✓ Cross-validation
# ✓ Final model selection
```

### Step 4: FDI-Specific Analysis
```bash
# You'll create:
notebooks/4_fdi_analysis.ipynb

# Which will:
# ✓ Compare performance by FDI status
# ✓ Analyze High/Medium/Low FDI stocks
# ✓ Sector-specific patterns
# ✓ Risk impact analysis
```

---

## 💡 Key Advantages of Multi-Algorithm Approach

### 1. **Robustness**
- Tests across complexity spectrum (simple to complex)
- Identifies if complex models add value

### 2. **Interpretability**
- Baseline & ML models provide feature importance
- GNN models can be understood through attention weights

### 3. **Practical Application**
- Some algorithms may be better for production (speed, simplicity)
- Others better for research (accuracy, flexibility)

### 4. **Research Contribution**
- Not just "applying GNNs to Vietnam stocks"
- Scientifically compares which approach works best
- Publishable finding: "For Vietnamese FDI stocks, [Model X] outperforms more complex alternatives by [Y]%"

---

## 📋 Algorithm Selection Logic

### When to Choose Each:

**Choose Historical Mean if:**
- Goal: Simple reference baseline
- Fast, no training required
- Use: Comparison point only

**Choose ARIMA if:**
- Data is stationary
- Traditional time-series domain
- Need interpretable model
- Good for: Financial data with trends

**Choose Random Forest if:**
- Best overall RMSE achieved by RF
- Non-linear patterns important
- Feature importance needed
- Fast training/inference required
- Recommended for production

**Choose LSTM if:**
- Best RMSE achieved by LSTM
- Long-term temporal dependencies crucial
- Can afford computation cost
- Good for: Complex temporal patterns

**Choose GRU if:**
- Performs similar to LSTM
- Faster training desired
- Less computation available
- Similar accuracy as LSTM

**Choose GNN if (Optional Enhancement):**
- Need to model stock correlations
- Graph structure important
- Want to capture sector relationships
- Research paper focus: GNN for stock prediction

---

## 🔍 Evaluation Metrics

Each model evaluated on:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **RMSE** | √(Σ(y-ŷ)²/n) | Lower is better, penalizes large errors |
| **MAE** | Σ\|y-ŷ\|/n | Average absolute error, interpretable |
| **MAPE** | Σ\|y-ŷ\|/y/n×100% | Percentage error, scale-independent |
| **R²** | 1 - SS_res/SS_tot | Proportion of variance explained |

---

## 📦 Dependencies

### Core
```bash
pip install numpy pandas scikit-learn
```

### Baseline
```bash
pip install statsmodels  # For ARIMA
```

### ML
```bash
pip install scikit-learn  # Already included above
```

### DL
```bash
pip install tensorflow keras
# or
pip install torch torchvision torchaudio
```

### All at once
```bash
pip install -r requirements.txt
pip install statsmodels tensorflow
```

---

## 📊 Expected Workflow

```
Week 1:
  ├─ Run notebooks/0_data_collection.ipynb (5 min)
  ├─ Run notebooks/1_data_preparation.ipynb (10 min)
  └─ Run notebooks/2_model_comparison.ipynb (30 min)
  
Result: Know which algorithm works best!

Week 2:
  ├─ Create notebooks/3_optimal_model.ipynb (fine-tune)
  ├─ Create notebooks/4_fdi_analysis.ipynb (FDI insights)
  └─ Create notebooks/5_visualization.ipynb (results)
  
Result: Complete analysis with FDI-specific findings
```

---

## 🎯 Research Questions Answered

By comparing these algorithms, you answer:

1. **Q: What's the optimal algorithm for Vietnamese FDI stock volatility?**
   - A: Run notebook 2 → Get comparison table → Recommendation

2. **Q: Does complexity add value?**
   - A: Compare Baseline vs ML vs DL RMSE improvement
   - If improvement < 5%, simpler model sufficient

3. **Q: What features matter for volatility?**
   - A: Random Forest provides feature importance

4. **Q: How do FDI stocks compare?**
   - A: Notebook 4 analyzes by FDI status and sector

5. **Q: Can we predict volatility accurately?**
   - A: MAPE < 5% = very good; MAPE < 10% = good; MAPE > 15% = needs improvement

---

## ✅ Checklist

- [x] Create model comparison framework (model_comparison.py)
- [x] Create comparison notebook (2_model_comparison.ipynb)
- [ ] Run data collection (0_data_collection.ipynb)
- [ ] Run feature preparation (1_data_preparation.ipynb)
- [ ] Run model comparison (2_model_comparison.ipynb)
- [ ] Analyze results and select optimal algorithm
- [ ] Create fine-tuning notebook (3_optimal_model.ipynb)
- [ ] Create FDI analysis notebook (4_fdi_analysis.ipynb)
- [ ] Create visualization notebook (5_visualization.ipynb)
- [ ] Write thesis/paper with findings

---

## 🎓 Next Step

Open and execute: **`notebooks/2_model_comparison.ipynb`**

This notebook will test all 4 algorithm groups and recommend the optimal one for your Vietnamese FDI stock volatility prediction task!

---

**Research Strategy**: Systematic comparison → Find optimal algorithm → Deep FDI-specific analysis → Publication-ready results 📊📈
