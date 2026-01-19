"""
Utility functions for data collection and processing
Adapted for Vietnamese stock market data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional
import time


def get_trading_days(start_date: str, end_date: str, country: str = 'VN') -> pd.DatetimeIndex:
    """
    Generate trading days for Vietnamese stock market
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        country: Country code (default: 'VN' for Vietnam)
    
    Returns:
        DatetimeIndex of trading days
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Remove Vietnamese holidays (simplified - add actual holiday calendar)
    vietnam_holidays = [
        '2024-01-01',  # New Year
        '2024-02-08', '2024-02-09', '2024-02-10', '2024-02-11', '2024-02-12',  # Tet
        '2024-04-18', '2024-04-19', '2024-04-20',  # Hung Kings
        '2024-04-30',  # Reunification Day
        '2024-05-01',  # Labor Day
        '2024-09-02',  # National Day
    ]
    
    holidays = pd.to_datetime(vietnam_holidays)
    trading_days = date_range[~date_range.isin(holidays)]
    
    return trading_days


def download_stock_data(ticker: str, start_date: str, end_date: str, 
                       source: str = 'vnstock') -> pd.DataFrame:
    """
    Download historical stock data for Vietnamese stocks
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        source: Data source ('vnstock', 'vndirect', or 'manual')
    
    Returns:
        DataFrame with OHLCV data
    """
    if source == 'vnstock':
        try:
            # Try using vnstock library
            from vnstock import stock_historical_data
            df = stock_historical_data(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date,
                resolution='1D',
                type='stock'
            )
            return df
        except ImportError:
            print(f"vnstock not installed. Using manual data for {ticker}")
            return generate_sample_data(ticker, start_date, end_date)
    else:
        return generate_sample_data(ticker, start_date, end_date)


def generate_sample_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate sample stock data for demonstration purposes
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with sample OHLCV data
    """
    np.random.seed(hash(ticker) % 2**32)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    num_days = len(dates)
    
    # Generate realistic price movements
    initial_price = np.random.uniform(50, 200)
    prices = [initial_price]
    
    for _ in range(num_days - 1):
        drift = 0.0002
        volatility = 0.02
        daily_return = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    # Create OHLCV data
    data = {
        'time': dates,
        'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
        'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 10000000, size=num_days)
    }
    
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    
    return df


def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calculate returns from price series
    
    Args:
        prices: Series of prices
        method: 'simple' or 'log'
    
    Returns:
        Series of returns
    """
    if method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        return prices.pct_change()


def calculate_volatility(returns: pd.Series, window: int = 20, 
                        annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility
    
    Args:
        returns: Series of returns
        window: Rolling window size
        annualize: Whether to annualize volatility
    
    Returns:
        Series of volatility values
    """
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize data
    
    Args:
        data: Array to normalize
        method: 'standard', 'minmax', or 'robust'
    
    Returns:
        Normalized array
    """
    if method == 'standard':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_adjacency_matrix(correlation_matrix: pd.DataFrame, 
                           threshold: float = 0.3) -> np.ndarray:
    """
    Create adjacency matrix from correlation matrix
    
    Args:
        correlation_matrix: Correlation matrix DataFrame
        threshold: Correlation threshold for edge creation
    
    Returns:
        Adjacency matrix as numpy array
    """
    adj_matrix = (np.abs(correlation_matrix.values) > threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
    
    return adj_matrix


def save_adjacency_matrix(adj_matrix: np.ndarray, filepath: str):
    """
    Save adjacency matrix to .npy file
    
    Args:
        adj_matrix: Adjacency matrix
        filepath: Path to save file
    """
    np.save(filepath, adj_matrix)
    print(f"Adjacency matrix saved to {filepath}")


def load_stock_list(filepath: str) -> pd.DataFrame:
    """
    Load stock list from CSV
    
    Args:
        filepath: Path to stock list CSV
    
    Returns:
        DataFrame with stock information
    """
    return pd.read_csv(filepath)


def get_fundamental_features(ticker: str) -> Dict:
    """
    Get fundamental features for a stock
    
    Args:
        ticker: Stock ticker
    
    Returns:
        Dictionary of fundamental features
    """
    # Placeholder - in practice, fetch from financial data API
    np.random.seed(hash(ticker) % 2**32)
    
    return {
        'ticker': ticker,
        'pe_ratio': np.random.uniform(5, 30),
        'pb_ratio': np.random.uniform(0.5, 5),
        'roe': np.random.uniform(0.05, 0.30),
        'debt_to_equity': np.random.uniform(0.1, 2.0),
        'dividend_yield': np.random.uniform(0, 0.08),
        'market_cap': np.random.uniform(1e9, 1e11),
        'beta': np.random.uniform(0.5, 2.0),
    }


def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate DataFrame has required columns and no critical missing values
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        True if valid, False otherwise
    """
    # Check columns exist
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        print(f"Missing columns: {missing}")
        return False
    
    # Check for excessive missing values
    missing_pct = df[required_columns].isnull().sum() / len(df)
    if (missing_pct > 0.1).any():
        print(f"Excessive missing values in: {missing_pct[missing_pct > 0.1].index.tolist()}")
        return False
    
    return True


def resample_to_daily(df: pd.DataFrame, date_column: str = 'time') -> pd.DataFrame:
    """
    Resample data to daily frequency
    
    Args:
        df: DataFrame with datetime index or column
        date_column: Name of date column
    
    Returns:
        Resampled DataFrame
    """
    df = df.copy()
    if date_column in df.columns:
        df = df.set_index(date_column)
    
    df.index = pd.to_datetime(df.index)
    
    # Resample to daily, forward fill missing values
    df_daily = df.resample('D').last()
    df_daily = df_daily.fillna(method='ffill')
    
    return df_daily.reset_index()


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test trading days
    trading_days = get_trading_days('2023-01-01', '2023-12-31')
    print(f"Number of trading days in 2023: {len(trading_days)}")
    
    # Test sample data generation
    sample_data = generate_sample_data('VNM', '2023-01-01', '2023-12-31')
    print(f"\nSample data shape: {sample_data.shape}")
    print(sample_data.head())
    
    print("\n✓ Utilities tested successfully!")
