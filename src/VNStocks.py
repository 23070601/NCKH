"""
VNStocks - Vietnamese Stock Data Collector
Collects price data and generates values.csv and adj.npy
Structure follows SP100AnalysisWithGNNs for scientific research
"""

import pandas as pd
import numpy as np
from typing import Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    download_stock_data,
    get_trading_days,
    create_adjacency_matrix,
    save_adjacency_matrix,
    load_stock_list,
)


class VNStocksDataset:
    """
    Dataset for Vietnamese FDI stocks
    Outputs: values.csv (Symbol/Date index + features), adj.npy (100x100 adjacency)
    """
    
    def __init__(self, 
                 stock_list_path: str,
                 start_date: str,
                 end_date: str,
                 raw_dir: str = '../data/raw',
                 processed_dir: str = '../data/processed'):
        """Initialize dataset"""
        self.stock_list_path = stock_list_path
        self.start_date = start_date
        self.end_date = end_date
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        
        # Load stock list
        self.stock_list = load_stock_list(stock_list_path)
        self.tickers = self.stock_list['ticker'].drop_duplicates().tolist()
        self.num_stocks = len(self.tickers)
        
        print(f"Initialized: {self.num_stocks} stocks, {start_date} to {end_date}")
        
        # Data containers
        self.price_data = {}
        self.correlation_matrix = None
        self.adjacency_matrix = None
    
    def collect_price_data(self, source: str = 'manual'):
        """Collect historical price data for all stocks"""
        print(f"\n[1/3] Collecting price data...")
        
        for i, ticker in enumerate(self.tickers, 1):
            try:
                df = download_stock_data(ticker, self.start_date, self.end_date, source)
                if df is not None and not df.empty:
                    self.price_data[ticker] = df
                    print(f"  [{i:3d}/{self.num_stocks}] {ticker}: ✓")
                else:
                    print(f"  [{i:3d}/{self.num_stocks}] {ticker}: (no data)")
            except Exception as e:
                print(f"  [{i:3d}/{self.num_stocks}] {ticker}: ✗")
        
        print(f"✓ Collected {len(self.price_data)}/{self.num_stocks} stocks")
    
    def process_and_save(self):
        """Process into values.csv and adj.npy"""
        print(f"\n[2/3] Processing data...")
        
        # Create values DataFrame with (Symbol, Date) index and features
        values_df = self._create_engineered_values()
        
        # Calculate correlations and adjacency
        self._calculate_correlation_matrix(values_df)
        # Use lower threshold for sample data; increase to 0.3 for real data
        self._create_adjacency_matrix(correlation_threshold=0.1)
        
        # Save
        print(f"\n[3/3] Saving...")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        values_path = os.path.join(self.processed_dir, 'values.csv')
        values_df.to_csv(values_path)
        print(f"  ✓ values.csv: {values_df.shape}")
        print(f"    Columns: {list(values_df.columns)}")
        
        adj_path = os.path.join(self.processed_dir, 'adj.npy')
        save_adjacency_matrix(self.adjacency_matrix, adj_path)
        print(f"  ✓ adj.npy: {self.adjacency_matrix.shape}")
        print(f"\n✓ Complete!")
    
    def _create_engineered_values(self) -> pd.DataFrame:
        """Create (Symbol, Date) multi-index DataFrame with engineered features"""
        all_dates = get_trading_days(self.start_date, self.end_date)
        data_list = []
        
        for ticker in self.tickers:
            if ticker not in self.price_data:
                continue
            
            stock_df = self.price_data[ticker].copy()
            stock_df['time'] = pd.to_datetime(stock_df['time'])
            stock_df = stock_df.set_index('time')
            stock_df = stock_df.reindex(all_dates).ffill().bfill()
            
            stock_df['Symbol'] = ticker
            stock_df = stock_df.reset_index()
            stock_df = stock_df.rename(columns={'index': 'Date', 'close': 'Close'})
            data_list.append(stock_df[['Symbol', 'Date', 'Close']])
        
        values_df = pd.concat(data_list, ignore_index=True)
        values_df = values_df.set_index(['Symbol', 'Date'])
        
        # Add features per stock
        values_df = values_df.groupby(level='Symbol', group_keys=False).apply(self._add_features)
        
        return values_df
    
    @staticmethod
    def _add_features(group):
        """Add engineered features: Close, NormClose, DailyLogReturn, ALR1W/2W/1M/2M, RSI, MACD"""
        group = group.sort_index()
        
        # NormClose: z-score normalized
        mean_price = group['Close'].mean()
        std_price = group['Close'].std()
        group['NormClose'] = (group['Close'] - mean_price) / (std_price + 1e-8)
        
        # DailyLogReturn: daily log return
        group['DailyLogReturn'] = np.log(group['Close'] / group['Close'].shift(1))
        
        # ALR (Annualized Log Returns): different periods
        group['ALR1W'] = np.log(group['Close'] / group['Close'].shift(5)) * (252.0 / 5.0)
        group['ALR2W'] = np.log(group['Close'] / group['Close'].shift(10)) * (252.0 / 10.0)
        group['ALR1M'] = np.log(group['Close'] / group['Close'].shift(21)) * (252.0 / 21.0)
        group['ALR2M'] = np.log(group['Close'] / group['Close'].shift(42)) * (252.0 / 42.0)
        
        # RSI: 14-period
        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        group['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD: 12/26 EMA
        ema12 = group['Close'].ewm(span=12).mean()
        ema26 = group['Close'].ewm(span=26).mean()
        group['MACD'] = ema12 - ema26
        
        return group.ffill().bfill()
    
    def _calculate_correlation_matrix(self, values_df: pd.DataFrame):
        """Calculate correlation from DailyLogReturn"""
        log_returns = values_df['DailyLogReturn'].unstack(level='Symbol').dropna()
        self.correlation_matrix = log_returns.corr()
    
    def _create_adjacency_matrix(self, correlation_threshold: float = 0.3):
        """Create adjacency from correlation"""
        adj = (np.abs(self.correlation_matrix.values) > correlation_threshold).astype(int)
        np.fill_diagonal(adj, 0)
        self.adjacency_matrix = adj
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'stocks': self.num_stocks,
            'collected': len(self.price_data),
            'date_range': f"{self.start_date} to {self.end_date}",
        }


def main():
    """Main entry point"""
    dataset = VNStocksDataset(
        stock_list_path='data/raw/fdi_stocks_list.csv',
        start_date='2022-01-01',
        end_date='2024-12-31',
        raw_dir='data/raw',
        processed_dir='data/processed'
    )
    
    dataset.collect_price_data(source='manual')
    dataset.process_and_save()
    print("\nSummary:", dataset.get_summary())


if __name__ == "__main__":
    main()
