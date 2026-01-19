# FDI Stock Market Analysis - Completion Guide

## Project: Predicting Volatility and Risk Level of Stock Prices of FDI Enterprises Listed in Vietnam

---

## ✅ COMPLETED COMPONENTS

### 1. **Stock List (100 FDI Companies)**
- **Location**: `data/fdi_stocks_list.csv`
- **Status**: ✅ Expanded to 100 stocks
- **Content**: Ticker, company name, sector, exchange, FDI status, market cap
- **Sectors**: Consumer, Finance, Energy, Materials, IT, Healthcare, Real Estate, Industrials
- **Exchanges**: HOSE (90 stocks), HNX (10 stocks)
- **FDI Levels**: High (5), Medium (42), Low (53)

### 2. **Data Collection Framework**
- **Files**: `src/VNStocks.py`, `src/utils.py`
- **Status**: ✅ Complete pipeline ready
- **Capabilities**:
  - Price data collection (sample or real via vnstock)
  - Fundamental metrics calculation
  - Adjacency matrix generation for GNN
  - Automatic data alignment and missing value handling

### 3. **Data Collection Notebook**
- **File**: `notebooks/0_data_collection.ipynb`
- **Status**: ✅ Ready to run
- **Outputs Generated**:
  - `stocks.csv` - Stock metadata (100 x 6)
  - `fundamentals.csv` - Financial metrics (PE, ROE, Beta, etc.)
  - `values.csv` - Aligned daily prices
  - `adj.npy` - Correlation-based adjacency matrix

---

## 📋 REMAINING STEPS (In Order)

### **Step 1: Run Data Collection Pipeline** (5-10 minutes)
```
Location: notebooks/0_data_collection.ipynb
Action: Execute all cells in order
```

**Process:**
1. Import libraries and setup paths
2. Load 100-stock list
3. Collect price data (currently sample data - 250 days each)
4. Collect fundamental features
5. Save CSV files and adjacency matrix

**Note**: First run uses sample data. To use real Vietnamese stock data:
```python
# Install vnstock
pip install vnstock

# In cell with data collection, change:
dataset.collect_price_data(source='vnstock')  # Instead of 'manual'
```

**Expected Outputs in `data/`:**
- ✅ stocks.csv (100 stocks)
- ✅ fundamentals.csv (PE, PB, ROE, Beta, etc.)
- ✅ values.csv (250 x 100 price matrix)
- ✅ adj.npy (100 x 100 correlation matrix)

---

### **Step 2: Feature Engineering & Volatility Calculation** (10-15 minutes)
```
Location: notebooks/1_data_preparation.ipynb
Action: Execute all cells in order (already scaffolded)
```

**Tasks to Complete:**
1. **Load Generated Data Files**
   - Import stocks.csv, fundamentals.csv, values.csv
   - Load adjacency matrix (adj.npy)

2. **Calculate Volatility Metrics** (Primary Target Variable)
   ```python
   # For each stock, calculate:
   - 20-day rolling volatility (annualized)
   - Volatility of volatility
   - Historical volatility bands
   ```

3. **Calculate Risk Metrics**
   ```python
   - Beta (market sensitivity)
   - Max drawdown
   - Value at Risk (VaR)
   - Conditional Value at Risk (CVaR)
   ```

4. **Engineer Additional Features**
   ```python
   - Returns (log and simple)
   - Price momentum indicators
   - Correlation stability
   - Sector-relative metrics
   - FDI status impact features
   ```

5. **Create Target Variables**
   ```python
   # Classification targets:
   - High volatility: σ > 75th percentile
   - Medium volatility: 25-75th percentile
   - Low volatility: σ < 25th percentile
   
   # Regression targets:
   - Predicted volatility at t+5, t+10 days
   ```

6. **Generate Train/Test Splits**
   ```python
   - Training: 2022-2023 (60%)
   - Validation: 2024 Q1-Q2 (20%)
   - Testing: 2024 Q3-Q4 (20%)
   ```

7. **Create GNN-Ready Datasets**
   - Node features: [fundamentals + recent prices + technical indicators]
   - Node labels: [volatility risk level]
   - Edge list: From adj.npy (correlation > 0.3 threshold)
   - Temporal sequences: Rolling 20-day windows

**Key Outputs:**
- `prepared_data.pkl` or HDF5 files with processed features
- `train_dataset.pkl`, `val_dataset.pkl`, `test_dataset.pkl`
- `feature_statistics.csv` - Mean, std, min, max for all features

---

### **Step 3: Build Graph Neural Network Models** (Requires PyTorch Geometric)
```
Estimated Time: 2-4 hours development
Key References: Professor's S&P 100 notebooks (2-8)
```

**Implementation Roadmap:**

#### **3a. Setup PyTorch Geometric Dataset**
```python
# Create custom PyTorch Geometric dataset class
- Nodes: 100 stocks
- Edges: Correlation-based (adj.npy)
- Node features: Financial + price features
- Labels: Volatility levels or values
```

#### **3b. Implement GNN Architecture Options**

**Option 1: Temporal Graph Convolutional Network (T-GCN)**
```python
# Best for: Volatility time-series prediction
# Combines:
- Spatial: Graph Convolution (stock relationships)
- Temporal: RNN/LSTM (price dynamics)

Architecture:
Input (100 x feature_dim) 
→ Graph Conv Layer 1 → ReLU
→ Graph Conv Layer 2 → ReLU
→ GRU/LSTM layers (temporal)
→ Dense output
```

**Option 2: Spatio-Temporal Graph CNN (STGCN)**
```python
# Best for: Multi-step ahead volatility forecast
# Combines temporal convolutions with graph convolutions

Architecture:
Input temporal graph (T=20, N=100)
→ ST-Conv blocks (3-4)
→ Temporal attention
→ Output predictions
```

**Option 3: Graph Attention Network (GAT)**
```python
# Best for: Understanding stock relationships
# Features: Attention-based edge weighting

Architecture:
Attention layers learn which stocks influence volatility
→ Multi-head attention (8 heads)
→ Edge weighting based on data
```

#### **3c. Training Pipeline**
```python
# Pseudocode:
for epoch in range(100):
    for batch in train_loader:
        # Get graph data with temporal windows
        graphs, labels = batch
        
        # Forward pass
        predictions = model(graphs)
        
        # Loss: Classification or Regression
        if task == "classification":
            loss = CrossEntropyLoss(predictions, labels)
        else:
            loss = MSELoss(predictions, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    val_metrics = evaluate(model, val_loader)
```

#### **3d. Evaluation Metrics**

**For Classification (High/Medium/Low Volatility):**
- Accuracy
- Precision/Recall/F1 (per class)
- Confusion matrix
- ROC-AUC

**For Regression (Volatility Prediction):**
- RMSE
- MAE
- MAPE
- Directional Accuracy (forecast sign correct)

---

### **Step 4: Risk-Specific Analysis for FDI Stocks** (NEW - Unique to Your Project)
```
Estimated Time: 2-3 hours
Key Differentiator: FDI-specific insights
```

#### **4a. FDI Status Impact Analysis**
```python
# Analyze volatility differences by FDI involvement:

# Separate models/analysis for:
1. High FDI stocks (VNM, SAB, VJC, PNJ, KDC)
   - Expected: More stable (foreign investment brings stability)
   
2. Medium FDI stocks (42 companies)
   - Expected: Moderate volatility
   
3. Low FDI stocks (53 companies)
   - Expected: Higher volatility

# Compare:
- Volatility levels
- Prediction accuracy
- GNN edge strength
```

#### **4b. Sector-Specific Risk Profiles**
```python
# By sector:
- Consumer: Defensive (lower vol expected)
- Finance: Systemic (correlation important)
- Energy: Commodity-linked (higher vol)
- Materials: Cyclical (sector clusters)
- IT: Growth (high vol potential)
- Real Estate: Sensitive to rates

# Create sector-specific models or weights
```

#### **4c. Cross-Border Risk Factors**
```python
# Unique to Vietnamese FDI stocks:
1. USD/VND exchange rate impact
2. Vietnamese macro indicators
3. Global commodity price correlation
4. China supply chain dependency
5. Trade policy effects
```

---

### **Step 5: Results and Visualization** (3-5 hours)

#### **5a. Model Performance Report**
```
Create summary including:
- Accuracy/RMSE on test set
- Per-stock prediction quality
- Sector performance comparison
- FDI status impact quantification
```

#### **5b. Visualizations**
```python
# Create figures for thesis/presentation:

1. Stock Network Graph
   - Nodes: 100 stocks
   - Edges: Correlations
   - Node color: Volatility level
   - Node size: Market cap

2. Volatility Heatmap
   - Time series of predicted vs actual
   - By sector
   - By FDI status

3. Feature Importance
   - Which features predict volatility best?
   - PCA visualization of stock embeddings

4. Risk Clusters
   - t-SNE plot of learned stock representations
   - Clusters by sector and FDI status

5. GNN Attention Weights
   - Which stock pairs are most correlated?
   - Edge strength visualization
```

#### **5c. Case Studies**
```
Analyze specific interesting cases:
- Stocks with high FDI but high volatility
- Outliers in sector or exchange
- Largest GNN edge weights (stock pairs)
- Model failure cases (worst predictions)
```

---

## 🛠️ TECHNICAL SETUP CHECKLIST

### **Required Packages** (Already in requirements.txt)
```bash
pip install -r requirements.txt
```

### **Additional for GNN Development**
```bash
pip install torch torchvision torchaudio
pip install torch-geometric

# Or conda:
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-geometric -c pyg
```

### **Optional - For Real Vietnamese Stock Data**
```bash
pip install vnstock yfinance
```

---

## 📁 FINAL PROJECT STRUCTURE

```
NCKH/
├── README.md                                    # Project overview
├── COMPLETION_GUIDE.md                          # This file
├── requirements.txt                             # Dependencies
├── data/
│   ├── fdi_stocks_list.csv                      # ✅ 100 stocks
│   ├── stocks.csv                               # 📊 Generated
│   ├── fundamentals.csv                         # 📊 Generated
│   ├── values.csv                               # 📊 Generated
│   └── adj.npy                                  # 📊 Generated
├── notebooks/
│   ├── 0_data_collection.ipynb                  # ✅ Complete
│   ├── 1_data_preparation.ipynb                 # ⏳ Ready to complete
│   ├── 2_gnn_model_training.ipynb               # 🔲 To create
│   ├── 3_fdi_analysis.ipynb                     # 🔲 To create (FDI-specific)
│   └── 4_results_visualization.ipynb            # 🔲 To create
├── src/
│   ├── VNStocks.py                              # ✅ Data pipeline
│   └── utils.py                                 # ✅ Utilities
└── models/
    └── [saved model files]                      # 📁 To create
```

---

## 📊 RESEARCH CONTRIBUTION SUMMARY

### **How Your Project Differs from Reference (S&P 100):**

1. **Geographic Focus**: Vietnamese market (vs US)
2. **FDI Specialization**: Analyzes foreign investment impact
3. **Risk Metric**: Volatility prediction (vs price forecasting)
4. **Challenges**:
   - Smaller, less liquid market than US
   - More volatile stocks
   - FDI dependency factor
   - Language/data availability
5. **Opportunities**:
   - Emerging market GNN application
   - Novel FDI-risk relationship
   - Cross-border portfolio optimization

### **Publishable Insights You Could Generate:**
- Impact of FDI status on stock volatility predictability
- Effectiveness of GNNs for emerging market volatility prediction
- Correlation structure changes during market stress
- Sector-specific GNN architectures for Vietnamese stocks

---

## ⏱️ ESTIMATED TIMELINE

| Phase | Task | Hours | Status |
|-------|------|-------|--------|
| 1 | Data Collection | 0.5 | ✅ Ready |
| 2 | Feature Engineering | 2 | ⏳ Next |
| 3a | GNN Dataset Creation | 1 | 🔲 To do |
| 3b | Model Development | 3 | 🔲 To do |
| 3c | Training & Tuning | 3 | 🔲 To do |
| 4 | FDI Analysis | 2 | 🔲 To do |
| 5 | Visualization & Report | 4 | 🔲 To do |
| **Total** | | **15.5** | |

---

## 🚀 IMMEDIATE NEXT STEPS

1. **Today/Tomorrow**:
   - [ ] Run `notebooks/0_data_collection.ipynb` (all cells)
   - [ ] Verify 4 output files generated in `data/`
   - [ ] Review generated data shapes and content

2. **Next 2-3 Days**:
   - [ ] Work through `notebooks/1_data_preparation.ipynb`
   - [ ] Calculate volatility metrics
   - [ ] Engineer features

3. **Week 2**:
   - [ ] Design GNN architecture
   - [ ] Create PyTorch Geometric dataset
   - [ ] Begin model training

4. **Week 3-4**:
   - [ ] Fine-tune models
   - [ ] Analyze FDI impact
   - [ ] Generate visualizations
   - [ ] Write results

---

## 📚 REFERENCE MATERIALS

### **Comparison with Professor's S&P 100 Project**
- Reference repository: https://github.com/timothewt/SP100AnalysisWithGNNs
- You are adapting notebooks 1-8 for Vietnamese FDI context
- Key difference: Focus on volatility + FDI impact

### **Key Papers on Temporal GNNs**
- Zhao et al. 2018: "Temporal Graph Neural Networks for Traffic Forecasting" (T-GCN)
- Li et al. 2017: "Diffusion Convolutional Recurrent Neural Networks"
- Zhu et al. 2021: "A3T-GCN: Attention Temporal Graph Convolutional Network"

### **Vietnamese Stock Market Resources**
- vnstock: Python library for Vietnamese stock data
- HOSE/HNX official websites: www.hsx.vn, www.hnx.vn
- Macro indicators: VN Index, VNEconomics

---

## ✨ GOOD LUCK WITH YOUR RESEARCH!

You've already completed the hardest part (data collection setup). Now execute the pipeline and watch your GNN learn to predict FDI stock volatility! 📈

For questions or issues, refer back to this guide or the in-notebook documentation.

**Key Success Factors:**
- ✅ Complete data pipeline daily (don't skip to modeling)
- ✅ Explore your data before modeling (EDA!)
- ✅ Test baseline models before complex GNNs
- ✅ Validate FDI hypothesis with statistical tests
- ✅ Document everything for your thesis

Happy researching! 🎓📊
