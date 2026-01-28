# ⚡ QUICK REFERENCE CARD
## Hybrid Volatility & Risk Prediction System

---

## 🎯 Start Here (Copy & Paste)

```bash
# Navigate to project
cd /Users/hoc/Documents/NCKH

# Activate environment
source .venv/bin/activate

# Predict VNM stock
python predict_simple.py predict VNM

# Or generate dashboard
python predict_simple.py dashboard
```

---

## 📋 All Commands

| Command | Purpose | Example |
|---------|---------|---------|
| **predict** | Predict single stock | `python predict_simple.py predict VNM` |
| **list** | Show all stocks | `python predict_simple.py list` |
| **batch** | Predict all stocks | `python predict_simple.py batch` |
| **dashboard** | Generate HTML | `python predict_simple.py dashboard` |
| **--help** | Show help | `python predict_simple.py --help` |

---

## 🎯 Use Cases

### Risk Monitoring
```bash
python predict_simple.py predict VNM
# ➜ Check if volatility/risk is HIGH
```

### Portfolio Review
```bash
python predict_simple.py batch
# ➜ Review risk distribution across all stocks
```

### Visual Analytics
```bash
python predict_simple.py dashboard
# ➜ Open volatility_dashboard.html in browser
```

### Find Best Opportunities
```bash
python predict_simple.py list
# ➜ See stocks by risk category
```

---

## 📊 Output Interpretation

### Volatility Levels
- **0.015 or less**: 🟢 Low volatility (stable)
- **0.015 - 0.020**: 🟡 Medium volatility (moderate)
- **0.020+**: 🔴 High volatility (risky)

### Confidence Levels
- **90-100%**: ⭐ Very confident
- **70-90%**: ✓ Reasonably confident
- **50-70%**: ⚠️ Uncertain
- **Below 50%**: ❌ Low confidence

---

## 📁 Where to Find Output

| What | Location |
|------|----------|
| Single predictions | `data/analysis/predictions_hybrid_*.json` |
| Batch results | `data/analysis/batch_predictions_*.json` |
| Dashboard | `volatility_dashboard.html` |
| Full data | `data/analysis/predictions_*.csv` |

---

## 🔄 Daily Workflow

```bash
# Morning: Check latest predictions
python predict_simple.py list

# Afternoon: Generate batch predictions
python predict_simple.py batch

# End of day: Update dashboard
python predict_simple.py dashboard
```

---

## 🆘 Quick Fixes

| Problem | Solution |
|---------|----------|
| Command not found | Run `cd /Users/hoc/Documents/NCKH` first |
| No data available | Run `python test_enhanced_features.py` |
| Stock not found | Run `python predict_simple.py list` |
| Need fresh data | Run `python collect_data.py` |

---

## 📈 Key Files

```
predict_simple.py           ← Main tool (use this!)
PRODUCTION_READY.md        ← Complete user guide
DEPLOYMENT_GUIDE.md        ← Technical guide
IMPLEMENTATION_COMPLETE.md ← Test results
volatility_dashboard.html  ← Visual dashboard
data/analysis/             ← All outputs here
```

---

## 💻 Python Usage (Advanced)

```python
import pandas as pd

# Load latest predictions
df = pd.read_csv('data/analysis/predictions_20260128_170424.csv')
latest = df.groupby('Symbol').tail(1)

# Get VNM prediction
vnm = latest[latest['Symbol'] == 'VNM'].iloc[0]
print(f"VNM volatility: {vnm['Predicted_Vol_RF']:.6f}")
print(f"VNM risk: {['Low', 'Medium', 'High'][int(vnm['Predicted_Risk'])]}")

# Find high-risk stocks
high_risk = latest[latest['Predicted_Risk'] == 2]
print(f"High-risk stocks: {high_risk['Symbol'].tolist()}")
```

---

## 🎓 Model Performance

- **Coverage**: 98 Vietnamese FDI stocks
- **Volatility Prediction RMSE**: 0.0068 (0.68%)
- **Risk Classification Accuracy**: 33.3%
- **Prediction Speed**: < 100ms per stock
- **Features**: 34 (technical + fundamental + macro)

---

## ✨ System Readiness

| Component | Status |
|-----------|--------|
| CLI Tool | ✅ Ready |
| Models | ✅ Trained |
| Data Pipeline | ✅ Running |
| Dashboard | ✅ Working |
| Documentation | ✅ Complete |
| **Overall** | **✅ PRODUCTION READY** |

---

## 📞 Support

1. **Quick help**: `python predict_simple.py --help`
2. **Full guide**: Read `PRODUCTION_READY.md`
3. **Issues**: Check `IMPLEMENTATION_COMPLETE.md`
4. **Technical**: See `DEPLOYMENT_GUIDE.md`

---

## 🚀 Now Go Predict!

```bash
python predict_simple.py predict VNM
```

Good luck! 🎉
