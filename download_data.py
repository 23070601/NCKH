import os
import sys
import glob
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def download_data(tickers: list, start_date: str, end_date: str, output_dir: str = 'data/raw'):
    print("=" * 70)
    print("DOWNLOADING STOCK DATA")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nDownloading data for {len(tickers)} symbols from {start_date} to {end_date}...")
    
    all_data = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            df['Symbol'] = ticker
            all_data.append(df)
            print(f"  ✓ {ticker}: {len(df)} records")
        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=False)
        
        output_file = os.path.join(output_dir, f"stock_data_{datetime.now().strftime('%Y%m%d')}.csv")
        combined.to_csv(output_file)
        
        print(f"\n✓ Downloaded {len(combined)} total records")
        print(f"✓ Saved to: {output_file}")
        
        return output_file
    else:
        print("\n✗ No data downloaded")
        return None


def main():
    tickers = ['VNM', 'HPG', 'TCB', 'BID', 'FPT', 'MWG', 'CTG', 'GVR']
    start_date = '2020-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    download_data(tickers, start_date, end_date)
    
    print("\n" + "=" * 70)
    print("✅ DATA DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
