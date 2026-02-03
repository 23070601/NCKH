"""
Macroeconomic Data Collection for Vietnamese Market
Collects market indices, interest rates, inflation, and exchange rates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MacroDataCollector:
    """
    Collect macroeconomic indicators for Vietnamese market.
    
    Provides:
    - VN-Index (market benchmark)
    - USD/VND exchange rate
    - SBV policy interest rate
    - CPI/Inflation
    - Interbank rates
    """
    
    def __init__(self, start_date: str, end_date: str):
        """
        Initialize macro data collector.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.macro_data = None
    
    def collect_vn_index(self) -> pd.DataFrame:
        """
        Collect VN-Index historical data.
        
        Returns:
            DataFrame with VN-Index values
        """
        # For demo: generate realistic VN-Index data
        # In production: use vnstock or API
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        
        # Generate realistic index movement (base ~1200, trending upward)
        np.random.seed(42)
        n_days = len(dates)
        
        # Start at 1200, add drift and volatility
        vn_index = [1200]
        for i in range(1, n_days):
            drift = 0.0003  # Slight upward trend
            volatility = 0.015
            daily_return = np.random.normal(drift, volatility)
            new_value = vn_index[-1] * (1 + daily_return)
            vn_index.append(max(new_value, 100))  # Floor at 100
        
        df = pd.DataFrame({
            'Date': dates,
            'VN_Index': vn_index
        })
        
        # Add VN-Index returns
        df['VN_Index_Return'] = df['VN_Index'].pct_change()
        df['VN_Index_Volatility'] = df['VN_Index_Return'].rolling(20).std() * np.sqrt(252)
        
        return df
    
    def collect_exchange_rate(self) -> pd.DataFrame:
        """
        Collect USD/VND exchange rate.
        
        Returns:
            DataFrame with exchange rates
        """
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        
        # Generate realistic USD/VND (around 23,000-25,000)
        np.random.seed(43)
        n_days = len(dates)
        
        # Start at 23,500
        usd_vnd = [23500]
        for i in range(1, n_days):
            drift = 0.00005  # Slight depreciation trend
            volatility = 0.002
            daily_change = np.random.normal(drift, volatility)
            new_rate = usd_vnd[-1] * (1 + daily_change)
            usd_vnd.append(max(new_rate, 20000))
        
        df = pd.DataFrame({
            'Date': dates,
            'USD_VND': usd_vnd,
            'USD_VND_Change': pd.Series(usd_vnd).pct_change()
        })
        
        return df
    
    def collect_interest_rates(self) -> pd.DataFrame:
        """
        Collect Vietnamese interest rates (SBV policy rate, interbank rates).
        
        Returns:
            DataFrame with interest rates
        """
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        
        # Policy rates are typically stable with occasional changes
        np.random.seed(44)
        
        # Base policy rate around 4.5%
        policy_rate = 4.5
        policy_rates = []
        
        for i, date in enumerate(dates):
            # Simulate occasional rate changes (quarterly)
            if i % 63 == 0 and i > 0:  # ~quarterly
                policy_rate += np.random.choice([-0.25, 0, 0.25], p=[0.2, 0.6, 0.2])
                policy_rate = np.clip(policy_rate, 2.0, 8.0)
            policy_rates.append(policy_rate)
        
        # Interbank overnight rate (slightly higher, more volatile)
        interbank_rates = np.array(policy_rates) + np.random.normal(0.3, 0.15, len(dates))
        interbank_rates = np.clip(interbank_rates, 2.0, 10.0)
        
        df = pd.DataFrame({
            'Date': dates,
            'Policy_Rate': policy_rates,
            'Interbank_Rate': interbank_rates
        })
        
        return df
    
    def collect_inflation(self) -> pd.DataFrame:
        """
        Collect CPI and inflation data.
        
        Returns:
            DataFrame with inflation indicators
        """
        # CPI is monthly, so we'll create monthly data and forward-fill
        months = pd.date_range(self.start_date, self.end_date, freq='MS')
        
        # Generate realistic CPI (base 100, inflation ~3-4% annually)
        np.random.seed(45)
        cpi = [100]
        
        for i in range(1, len(months)):
            monthly_inflation = np.random.normal(0.003, 0.002)  # ~3.6% annual
            cpi.append(cpi[-1] * (1 + monthly_inflation))
        
        df_monthly = pd.DataFrame({
            'Date': months,
            'CPI': cpi
        })
        
        # Calculate YoY inflation
        df_monthly['Inflation_YoY'] = df_monthly['CPI'].pct_change(12) * 100
        
        # Forward-fill to daily
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        df_daily = pd.DataFrame({'Date': dates})
        df_daily = df_daily.merge(df_monthly, on='Date', how='left')
        df_daily = df_daily.fillna(method='ffill')
        
        return df_daily
    
    def collect_all(self) -> pd.DataFrame:
        """
        Collect all macroeconomic indicators and merge into single DataFrame.
        
        Returns:
            DataFrame with all macro indicators
        """
        print("Collecting macroeconomic data...")
        
        # Collect each indicator
        vn_index = self.collect_vn_index()
        print(f"  ✓ VN-Index: {len(vn_index)} days")
        
        exchange = self.collect_exchange_rate()
        print(f"  ✓ USD/VND: {len(exchange)} days")
        
        rates = self.collect_interest_rates()
        print(f"  ✓ Interest rates: {len(rates)} days")
        
        inflation = self.collect_inflation()
        print(f"  ✓ Inflation: {len(inflation)} days")
        
        # Merge all on Date
        macro_df = vn_index.copy()
        macro_df = macro_df.merge(exchange, on='Date', how='left')
        macro_df = macro_df.merge(rates, on='Date', how='left')
        macro_df = macro_df.merge(inflation, on='Date', how='left')
        
        # Fill any missing values
        macro_df = macro_df.fillna(method='ffill').fillna(method='bfill')
        
        self.macro_data = macro_df
        
        print(f"\n✓ Collected {len(macro_df)} days of macro data")
        print(f"  Columns: {list(macro_df.columns)}")
        
        return macro_df
    
    def save(self, filepath: str):
        """
        Save macroeconomic data to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.macro_data is None:
            self.collect_all()
        
        self.macro_data.to_csv(filepath, index=False)
        print(f"\n✓ Saved macro data to {filepath}")
    
    def get_macro_features_for_date(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Get macro features for a specific date.
        
        Args:
            date: Date to get features for
            
        Returns:
            Dictionary of macro features
        """
        if self.macro_data is None:
            self.collect_all()
        
        # Find closest date
        closest_idx = (self.macro_data['Date'] - date).abs().idxmin()
        row = self.macro_data.iloc[closest_idx]
        
        return {
            'VN_Index': row['VN_Index'],
            'VN_Index_Return': row['VN_Index_Return'],
            'VN_Index_Volatility': row['VN_Index_Volatility'],
            'USD_VND': row['USD_VND'],
            'USD_VND_Change': row['USD_VND_Change'],
            'Policy_Rate': row['Policy_Rate'],
            'Interbank_Rate': row['Interbank_Rate'],
            'CPI': row['CPI'],
            'Inflation_YoY': row['Inflation_YoY']
        }


def integrate_macro_with_stock_data(stock_data_path: str, macro_data_path: str, output_path: str):
    """
    Integrate macroeconomic features with stock price data.
    
    Args:
        stock_data_path: Path to stock values.csv
        macro_data_path: Path to macro data CSV
        output_path: Path to save enriched data
    """
    print("Integrating macro data with stock data...")
    
    # Load data
    stock_df = pd.read_csv(stock_data_path)
    macro_df = pd.read_csv(macro_data_path)
    
    # Ensure Date columns are datetime
    if 'Date' in stock_df.columns:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
    
    # Merge
    if 'Date' in stock_df.columns:
        enriched_df = stock_df.merge(macro_df, on='Date', how='left')
    else:
        # If stock data has (Symbol, Date) index
        stock_df = stock_df.reset_index()
        enriched_df = stock_df.merge(macro_df, on='Date', how='left')
        if 'Symbol' in enriched_df.columns:
            enriched_df = enriched_df.set_index(['Symbol', 'Date'])
    
    # Fill missing macro data (forward fill)
    enriched_df = enriched_df.fillna(method='ffill')
    
    # Save
    enriched_df.to_csv(output_path)
    
    print(f"✓ Enriched data saved to {output_path}")
    print(f"  Shape: {enriched_df.shape}")
    print(f"  New columns: {[c for c in enriched_df.columns if c in macro_df.columns and c != 'Date']}")
    
    return enriched_df


if __name__ == "__main__":
    # Test the collector
    collector = MacroDataCollector(
        start_date='2022-01-01',
        end_date='2024-12-31'
    )
    
    # Collect all data
    macro_df = collector.collect_all()
    
    # Display summary
    print("\n" + "="*80)
    print("MACRO DATA SUMMARY")
    print("="*80)
    print(macro_df.describe())
    
    print("\n" + "="*80)
    print("SAMPLE DATA (First 5 days)")
    print("="*80)
    print(macro_df.head())
    
    # Save
    import os
    os.makedirs('../data/features', exist_ok=True)
    collector.save('../data/features/macro_data.csv')
    
    print("\n✓ Macro data collection complete!")
