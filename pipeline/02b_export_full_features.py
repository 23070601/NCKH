"""
Export ALL features into single comprehensive CSV files.
Creates both raw and processed versions.
"""
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

def export_full_features():
    """Export all features into comprehensive CSV files."""
    
    # Read raw values
    values_path = 'data/raw/values.csv'
    if not os.path.exists(values_path):
        print(f"❌ {values_path} not found")
        return
    
    print("Loading raw values...")
    df = pd.read_csv(values_path)
    
    # Sort by Date and Symbol
    df = df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
    
    # Save raw features (all features, wide format)
    output_raw = 'data/features/all_features_raw.csv'
    df.to_csv(output_raw, index=False)
    print(f"✓ Exported raw features: {output_raw}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Create processed version (normalized)
    print("\nCreating processed version (normalized)...")
    df_processed = df.copy()
    
    # Normalize numeric columns (excluding Date, Symbol)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col in df_processed.columns:
            mean = df_processed[col].mean()
            std = df_processed[col].std()
            if std > 0:
                df_processed[col] = (df_processed[col] - mean) / std
    
    output_processed = 'data/features/all_features_processed.csv'
    df_processed.to_csv(output_processed, index=False)
    print(f"✓ Exported processed features: {output_processed}")
    print(f"  Shape: {df_processed.shape}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Unique symbols: {df['Symbol'].nunique()}")
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Features: {', '.join(numeric_cols[:10])}...")
    
    return df, df_processed


if __name__ == '__main__':
    print("="*60)
    print("EXPORT FULL FEATURE DATASET")
    print("="*60)
    
    df_raw, df_processed = export_full_features()
    
    print("\n✅ Done! Files created:")
    print("   - data/features/all_features_raw.csv")
    print("   - data/features/all_features_processed.csv")
