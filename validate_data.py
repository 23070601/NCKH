#!/usr/bin/env python3
"""Validate and report on data quality"""
import pandas as pd
import numpy as np

values = pd.read_csv('data/processed/values.csv', index_col=[0, 1])
adj = np.load('data/processed/adj.npy')

print("\nDATA QUALITY REPORT")
print("=" * 70)

# Structure
print("\n[STRUCTURE]")
print(f"  values.csv: {values.shape[0]:,} rows × {values.shape[1]} features")
print(f"  Unique stocks: {values.index.get_level_values('Symbol').nunique()}")
print(f"  Unique dates: {values.index.get_level_values('Date').nunique()}")
print(f"  adj.npy: {adj.shape[0]} × {adj.shape[1]} adjacency matrix")

# Missing values
print("\n[DATA INTEGRITY]")
total_null = values.isnull().sum().sum()
print(f"  Total null values: {total_null}")
if total_null == 0:
    print("  ✓ No missing data")

# Features
print("\n[FEATURES]")
for col in values.columns:
    print(f"  - {col}")

# Basic stats
print("\n[STATISTICS]")
print(f"  Close price range: {values['Close'].min():.2f} - {values['Close'].max():.2f}")
print(f"  Daily returns: μ={values['DailyLogReturn'].mean():.4f}, σ={values['DailyLogReturn'].std():.4f}")
print(f"  RSI range: {values['RSI'].min():.1f} - {values['RSI'].max():.1f}")
print(f"  MACD range: {values['MACD'].min():.4f} - {values['MACD'].max():.4f}")

# Date info
dates = pd.to_datetime(values.index.get_level_values('Date'))
print("\n[DATE RANGE]")
print(f"  Start: {dates.min().date()}")
print(f"  End: {dates.max().date()}")
print(f"  Trading days: {len(dates.unique())}")

# Adjacency
print("\n[ADJACENCY MATRIX]")
print(f"  Non-zero edges: {np.count_nonzero(adj):,}")
print(f"  Density: {np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.4f}")
print(f"  Symmetric: {np.allclose(adj, adj.T)}")

print("\n" + "=" * 70)
print("✓ Data validation complete")
