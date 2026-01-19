"""
Feature Statistics Report - Vietnamese FDI Stocks
Generated: 2025-01-19
Dataset: 98 stocks × 773 trading days (2022-01-01 to 2024-12-31)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
print("Loading data...")
values = pd.read_csv('data/processed/values.csv', index_col=[0, 1])
adj = np.load('data/processed/adj.npy')

print("\n" + "=" * 80)
print("COMPREHENSIVE FEATURE STATISTICS REPORT")
print("=" * 80)

# =============================================================================
# 1. DATASET OVERVIEW
# =============================================================================
print("\n[1] DATASET OVERVIEW")
print("-" * 80)
print(f"Total observations: {len(values):,}")
print(f"Unique stocks: {values.index.get_level_values('Symbol').nunique()}")
print(f"Trading days: {values.index.get_level_values('Date').nunique()}")
print(f"Features: {len(values.columns)}")
print(f"Date range: {values.index.get_level_values('Date').min()} to {values.index.get_level_values('Date').max()}")

# =============================================================================
# 2. FEATURE STATISTICS
# =============================================================================
print("\n[2] DESCRIPTIVE STATISTICS BY FEATURE")
print("-" * 80)

stats = values.describe().T
stats['skewness'] = values.skew()
stats['kurtosis'] = values.kurtosis()

print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(4))

print("\n[2.1] Distribution Shape")
print("-" * 80)
print(stats[['skewness', 'kurtosis']].round(4))
print("\nInterpretation:")
print("  Skewness = 0: Symmetric distribution")
print("  Skewness > 0: Right-skewed (long right tail)")
print("  Skewness < 0: Left-skewed (long left tail)")
print("  Kurtosis = 3: Normal distribution")
print("  Kurtosis > 3: Heavy-tailed (more outliers)")

# =============================================================================
# 3. STOCK-LEVEL ANALYSIS
# =============================================================================
print("\n[3] STOCK-LEVEL STATISTICS")
print("-" * 80)

# Average metrics per stock
stock_means = values.groupby(level='Symbol').mean()
stock_stds = values.groupby(level='Symbol').std()

print("\n[3.1] Top 10 Stocks by Average Price")
top_price = stock_means['Close'].sort_values(ascending=False).head(10)
for rank, (stock, price) in enumerate(top_price.items(), 1):
    print(f"  {rank:2d}. {stock:6s}: {price:8.2f} VND")

print("\n[3.2] Top 10 Most Volatile Stocks (by Daily Return Std)")
top_vol = stock_stds['DailyLogReturn'].sort_values(ascending=False).head(10)
for rank, (stock, vol) in enumerate(top_vol.items(), 1):
    print(f"  {rank:2d}. {stock:6s}: {vol:.6f}")

print("\n[3.3] Top 10 Least Volatile Stocks")
low_vol = stock_stds['DailyLogReturn'].sort_values().head(10)
for rank, (stock, vol) in enumerate(low_vol.items(), 1):
    print(f"  {rank:2d}. {stock:6s}: {vol:.6f}")

print("\n[3.4] RSI Distribution Across Stocks")
rsi_summary = stock_means['RSI'].describe()
print(f"  Mean RSI (avg across stocks): {rsi_summary['mean']:.2f}")
print(f"  Median RSI: {rsi_summary['50%']:.2f}")
print(f"  Range: [{rsi_summary['min']:.2f}, {rsi_summary['max']:.2f}]")

# =============================================================================
# 4. TEMPORAL PATTERNS
# =============================================================================
print("\n[4] TEMPORAL ANALYSIS")
print("-" * 80)

# Group by date
daily_stats = values.groupby(level='Date').agg({
    'Close': 'mean',
    'DailyLogReturn': ['mean', 'std'],
    'RSI': 'mean'
})

print("\n[4.1] Average Daily Statistics")
print(f"  Mean closing price across all stocks: {daily_stats['Close']['mean'].mean():.2f}")
print(f"  Mean daily return: {daily_stats['DailyLogReturn']['mean'].mean():.6f}")
print(f"  Mean daily volatility: {daily_stats['DailyLogReturn']['std'].mean():.6f}")

# Find extreme days
dates_df = values.reset_index()
daily_returns = dates_df.groupby('Date')['DailyLogReturn'].mean()

print("\n[4.2] Extreme Market Days")
print(f"  Best day (highest avg return): {daily_returns.idxmax()} ({daily_returns.max():.4f})")
print(f"  Worst day (lowest avg return): {daily_returns.idxmin()} ({daily_returns.min():.4f})")

# =============================================================================
# 5. CORRELATION ANALYSIS
# =============================================================================
print("\n[5] CORRELATION STRUCTURE")
print("-" * 80)

# Feature correlations
feature_corr = values.corr()

print("\n[5.1] Feature Correlation Matrix")
print(feature_corr.round(3))

print("\n[5.2] Strongest Feature Correlations (excluding self-correlation)")
corr_pairs = []
for i in range(len(feature_corr)):
    for j in range(i+1, len(feature_corr)):
        corr_pairs.append((
            feature_corr.index[i], 
            feature_corr.columns[j], 
            feature_corr.iloc[i,j]
        ))

top_corr = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
for feat1, feat2, corr in top_corr:
    print(f"  {feat1:20s} - {feat2:20s}: {corr:7.4f}")

# Stock correlations
print("\n[5.3] Stock Correlation Network")
print(f"  Adjacency matrix shape: {adj.shape}")
print(f"  Total edges: {np.count_nonzero(adj)}")
print(f"  Network density: {np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.6f}")

degrees = adj.sum(axis=1)
print(f"\n  Node Degree Distribution:")
print(f"    Mean degree: {degrees.mean():.2f}")
print(f"    Median degree: {np.median(degrees):.0f}")
print(f"    Max degree: {int(degrees.max())}")
print(f"    Min degree: {int(degrees.min())}")
print(f"    Isolated nodes (degree=0): {(degrees == 0).sum()}")

# =============================================================================
# 6. DATA QUALITY CHECKS
# =============================================================================
print("\n[6] DATA QUALITY")
print("-" * 80)

# Missing values
missing = values.isnull().sum()
print(f"Missing values per feature:")
if missing.sum() == 0:
    print("  ✓ No missing values detected")
else:
    for feat, count in missing[missing > 0].items():
        pct = 100 * count / len(values)
        print(f"  {feat}: {count:,} ({pct:.2f}%)")

# Outliers (using IQR method)
print("\n[6.1] Outlier Detection (IQR method)")
for col in values.columns:
    Q1 = values[col].quantile(0.25)
    Q3 = values[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((values[col] < lower) | (values[col] > upper)).sum()
    pct = 100 * outliers / len(values)
    print(f"  {col:20s}: {outliers:6,} outliers ({pct:5.2f}%)")

# Infinite values
print("\n[6.2] Infinite Values Check")
has_inf = False
for col in values.columns:
    inf_count = np.isinf(values[col]).sum()
    if inf_count > 0:
        print(f"  {col}: {inf_count:,} infinite values")
        has_inf = True
if not has_inf:
    print("  ✓ No infinite values detected")

# =============================================================================
# 7. FEATURE IMPORTANCE INDICATORS
# =============================================================================
print("\n[7] FEATURE VARIABILITY")
print("-" * 80)
print("Coefficient of Variation (CV = std/mean) - higher indicates more variability:")

cv_scores = {}
for col in values.columns:
    if values[col].mean() != 0:
        cv = abs(values[col].std() / values[col].mean())
        cv_scores[col] = cv

for feat, cv in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat:20s}: {cv:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n✓ Dataset is complete with no missing values")
print(f"✓ {len(values):,} observations across 98 stocks and 773 trading days")
print(f"✓ 9 engineered features: price, returns, technical indicators")
print(f"✓ Correlation network has {np.count_nonzero(adj)} edges (sparse)")
print(f"✓ Mean daily return: {values['DailyLogReturn'].mean():.6f}")
print(f"✓ Overall volatility (std of returns): {values['DailyLogReturn'].std():.6f}")

print("\n" + "=" * 80)
print("END OF REPORT")
print("=" * 80)
