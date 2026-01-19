"""
VNStocks - Vietnamese Stock Data Collector
Similar to SP100Stocks.py but for Vietnamese FDI companies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os
import sys

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    download_stock_data,
    get_fundamental_features,
    calculate_returns,
    calculate_volatility,
    create_adjacency_matrix,
    save_adjacency_matrix,
    load_stock_list,
    validate_data,
    get_trading_days
)


class VNStocksDataset:
    """
    Dataset class for Vietnamese FDI stocks
    Collects and processes stock data for GNN analysis
    """
    
    def __init__(self, 
                 stock_list_path: str,
                 start_date: str,
                 end_date: str,
                 data_dir: str = '../data'):
        """
        Initialize dataset
        
        Args:
            stock_list_path: Path to CSV with stock tickers
            start_date: Start date for data collection
            end_date: End date for data collection
            data_dir: Directory to save processed data
        """
        self.stock_list_path = stock_list_path
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        
        # Load stock list
        self.stock_list = load_stock_list(stock_list_path)
        self.tickers = self.stock_list['ticker'].tolist()
        self.num_stocks = len(self.tickers)
        
        print(f"Initialized dataset with {self.num_stocks} stocks")
        print(f"Date range: {start_date} to {end_date}")
        
        # Data containers
        self.price_data = {}
        self.fundamentals_data = []
        self.returns_matrix = None
        self.correlation_matrix = None
        self.adjacency_matrix = None
    
    def collect_price_data(self, source: str = 'vnstock'):
        """
        Collect historical price data for all stocks
        
        Args:
            source: Data source ('vnstock', 'vndirect', or 'manual')
        """
        print(f"\nCollecting price data from {source}...")
        
        for i, ticker in enumerate(self.tickers, 1):
            print(f"[{i}/{self.num_stocks}] Downloading {ticker}...", end=' ')
            
            try:
                df = download_stock_data(ticker, self.start_date, self.end_date, source)
                
                if df is not None and not df.empty:
                    self.price_data[ticker] = df
                    print(f"✓ ({len(df)} days)")
                else:
                    print(f"✗ No data")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
            
        print(f"\nCollected data for {len(self.price_data)}/{self.num_stocks} stocks")
    
    def collect_fundamentals(self):
        """
        Collect fundamental data for all stocks
        """
        print("\nCollecting fundamental data...")
        
        for i, ticker in enumerate(self.tickers, 1):
            print(f"[{i}/{self.num_stocks}] {ticker}...", end=' ')
            
            try:
                fundamentals = get_fundamental_features(ticker)
                
                # Add stock info from stock list
                stock_info = self.stock_list[self.stock_list['ticker'] == ticker].iloc[0]
                fundamentals.update({
                    'name': stock_info['name'],
                    'sector': stock_info['sector'],
                    'exchange': stock_info['exchange'],
                    'fdi_status': stock_info['fdi_status']
                })
                
                self.fundamentals_data.append(fundamentals)
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"Collected fundamentals for {len(self.fundamentals_data)} stocks")
    
    def create_values_dataframe(self) -> pd.DataFrame:
        """
        Create unified values DataFrame (similar to values.csv in reference)
        All stocks aligned by date with closing prices
        
        Returns:
            DataFrame with dates and stock prices
        """
        print("\nCreating values DataFrame...")
        
        # Get all dates from the date range
        all_dates = get_trading_days(self.start_date, self.end_date)
        
        # Initialize DataFrame with dates
        values_df = pd.DataFrame({'Date': all_dates})
        
        # Add each stock's closing price
        for ticker in self.tickers:
            if ticker in self.price_data:
                stock_df = self.price_data[ticker].copy()
                stock_df['time'] = pd.to_datetime(stock_df['time'])
                stock_df = stock_df.set_index('time')
                
                # Reindex to match all dates, forward fill missing values
                stock_df = stock_df.reindex(all_dates, method='ffill')
                
                # Add to values dataframe
                values_df[ticker] = stock_df['close'].values
        
        # Fill any remaining NaNs with forward fill, then backward fill
        values_df = values_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Values DataFrame shape: {values_df.shape}")
        print(f"Date range: {values_df['Date'].min()} to {values_df['Date'].max()}")
        
        return values_df
    
    def calculate_correlation_matrix(self, values_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix from returns
        
        Args:
            values_df: DataFrame with stock prices
        
        Returns:
            Correlation matrix
        """
        print("\nCalculating correlation matrix...")
        
        # Calculate log returns
        price_columns = [col for col in values_df.columns if col != 'Date']
        returns = np.log(values_df[price_columns] / values_df[price_columns].shift(1))
        returns = returns.dropna()
        
        # Calculate correlation
        correlation_matrix = returns.corr()
        
        print(f"Correlation matrix shape: {correlation_matrix.shape}")
        print(f"Mean correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.4f}")
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def create_adjacency_matrix(self, correlation_threshold: float = 0.3):
        """
        Create adjacency matrix from correlation matrix
        
        Args:
            correlation_threshold: Threshold for edge creation
        """
        print(f"\nCreating adjacency matrix (threshold={correlation_threshold})...")
        
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not calculated. Run calculate_correlation_matrix first.")
        
        self.adjacency_matrix = create_adjacency_matrix(
            self.correlation_matrix, 
            threshold=correlation_threshold
        )
        
        num_edges = np.sum(self.adjacency_matrix) // 2  # Undirected graph
        density = num_edges / (self.num_stocks * (self.num_stocks - 1) / 2)
        
        print(f"Adjacency matrix shape: {self.adjacency_matrix.shape}")
        print(f"Number of edges: {num_edges}")
        print(f"Graph density: {density:.4f}")
    
    def save_all_data(self):
        """
        Save all processed data to files
        Creates: stocks.csv, fundamentals.csv, values.csv, adj.npy
        """
        print("\n" + "="*60)
        print("SAVING ALL DATA")
        print("="*60)
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 1. Save stocks.csv (stock list with metadata)
        stocks_path = os.path.join(self.data_dir, 'stocks.csv')
        self.stock_list.to_csv(stocks_path, index=False)
        print(f"✓ Saved stocks.csv ({len(self.stock_list)} stocks)")
        
        # 2. Save fundamentals.csv
        fundamentals_path = os.path.join(self.data_dir, 'fundamentals.csv')
        fundamentals_df = pd.DataFrame(self.fundamentals_data)
        fundamentals_df.to_csv(fundamentals_path, index=False)
        print(f"✓ Saved fundamentals.csv ({len(fundamentals_df)} records)")
        
        # 3. Save values.csv
        values_df = self.create_values_dataframe()
        values_path = os.path.join(self.data_dir, 'values.csv')
        values_df.to_csv(values_path, index=False)
        print(f"✓ Saved values.csv {values_df.shape}")
        
        # 4. Calculate and save correlation matrix
        self.calculate_correlation_matrix(values_df)
        
        # 5. Create and save adjacency matrix
        self.create_adjacency_matrix(correlation_threshold=0.3)
        adj_path = os.path.join(self.data_dir, 'adj.npy')
        save_adjacency_matrix(self.adjacency_matrix, adj_path)
        print(f"✓ Saved adj.npy {self.adjacency_matrix.shape}")
        
        print("\n" + "="*60)
        print("ALL DATA SAVED SUCCESSFULLY!")
        print("="*60)
        print(f"\nFiles created in {self.data_dir}:")
        print(f"  - stocks.csv: Stock metadata")
        print(f"  - fundamentals.csv: Fundamental features")
        print(f"  - values.csv: Daily closing prices")
        print(f"  - adj.npy: Adjacency matrix for GNN")
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of the dataset
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_stocks': self.num_stocks,
            'tickers': self.tickers,
            'date_range': (self.start_date, self.end_date),
            'data_collected': len(self.price_data),
            'fundamentals_collected': len(self.fundamentals_data),
        }
        
        if self.adjacency_matrix is not None:
            num_edges = np.sum(self.adjacency_matrix) // 2
            summary['num_edges'] = num_edges
            summary['graph_density'] = num_edges / (self.num_stocks * (self.num_stocks - 1) / 2)
        
        return summary


def main():
    """
    Main function to run data collection pipeline
    """
    print("="*60)
    print("VIETNAMESE FDI STOCKS DATA COLLECTION")
    print("="*60)
    
    # Configuration
    stock_list_path = '../data/fdi_stocks_list.csv'
    start_date = '2022-01-01'
    end_date = '2024-12-31'
    data_dir = '../data'
    
    # Initialize dataset
    dataset = VNStocksDataset(
        stock_list_path=stock_list_path,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir
    )
    
    # Collect all data
    dataset.collect_price_data(source='manual')  # Use 'vnstock' when library is available
    dataset.collect_fundamentals()
    
    # Save all data
    dataset.save_all_data()
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    summary = dataset.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n✓ Pipeline completed successfully!")
    print("\nNext steps:")
    print("  1. Review generated CSV files and adj.npy")
    print("  2. Run data preparation notebook")
    print("  3. Build GNN model for volatility prediction")


if __name__ == "__main__":
    main()
