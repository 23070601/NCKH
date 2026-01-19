#!/usr/bin/env python3
"""
Vietnamese FDI Stock Data Collection Pipeline
==============================================

This script collects and processes historical price data for 98 Vietnamese FDI stocks.

Outputs:
    - data/processed/values.csv: Time-series with (Symbol, Date) multi-index + 9 features
    - data/processed/adj.npy: Stock correlation adjacency matrix (98×98)

Usage:
    python collect_data.py
"""

import sys
import os
sys.path.insert(0, 'src')

from VNStocks import VNStocksDataset
import pandas as pd
import numpy as np

def main():
    print("=" * 80)
    print("VIETNAMESE FDI STOCK DATA COLLECTION")
    print("=" * 80)
    
    # Configuration
    config = {
        'stock_list_path': 'data/raw/fdi_stocks_list.csv',
        'start_date': '2022-01-01',
        'end_date': '2024-12-31',
        'raw_dir': 'data/raw',
        'processed_dir': 'data/processed',
        'source': 'manual'  # Change to 'vnstock' for real data
    }
    
    print(f"\n[CONFIG]")
    print(f"  Stock list: {config['stock_list_path']}")
    print(f"  Date range: {config['start_date']} to {config['end_date']}")
    print(f"  Data source: {config['source']}")
    print(f"  Output dir: {config['processed_dir']}")
    
    # Initialize dataset
    print(f"\n[STEP 1/3] Initializing dataset...")
    dataset = VNStocksDataset(
        stock_list_path=config['stock_list_path'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        raw_dir=config['raw_dir'],
        processed_dir=config['processed_dir']
    )
    
    # Collect price data
    print(f"\n[STEP 2/3] Collecting price data...")
    dataset.collect_price_data(source=config['source'])
    
    # Process and save
    print(f"\n[STEP 3/3] Processing and saving...")
    dataset.process_and_save()
    
    # Verify outputs
    print(f"\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    values_path = os.path.join(config['processed_dir'], 'values.csv')
    adj_path = os.path.join(config['processed_dir'], 'adj.npy')
    
    if os.path.exists(values_path):
        values = pd.read_csv(values_path, index_col=[0, 1])
        print(f"\n✓ values.csv")
        print(f"  Shape: {values.shape}")
        print(f"  Stocks: {values.index.get_level_values('Symbol').nunique()}")
        print(f"  Dates: {values.index.get_level_values('Date').nunique()}")
        print(f"  Features: {list(values.columns)}")
        print(f"  Size: {os.path.getsize(values_path) / 1024 / 1024:.2f} MB")
    
    if os.path.exists(adj_path):
        adj = np.load(adj_path)
        print(f"\n✓ adj.npy")
        print(f"  Shape: {adj.shape}")
        print(f"  Edges: {np.count_nonzero(adj)}")
        print(f"  Density: {np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.4f}")
        print(f"  Size: {os.path.getsize(adj_path) / 1024:.2f} KB")
    
    print(f"\n" + "=" * 80)
    print("✓ DATA COLLECTION COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"  1. Review: data/processed/values.csv and data/processed/adj.npy")
    print(f"  2. Analyze: jupyter notebook notebooks/1_data_preparation.ipynb")
    print(f"  3. Model: jupyter notebook notebooks/2_model_comparison.ipynb")

if __name__ == "__main__":
    main()
