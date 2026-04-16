#Data mining tools

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

URL = "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv"
YEARS = 10
ROOT = Path(__file__).resolve().parent.parent

def yf_assets_data(tickers: List[str], years: float | None = None) -> pd.DataFrame:
    # Calculate date range in years
    end_date = datetime.now()
    start_date = end_date - timedelta(days= years *365)  # Approx 30 years
    
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("-" * 50)


    for ticker in tqdm(tickers, desc="Downloading tickers", unit="ticker"):
        # Download data for all tickers
        print(f"Downloading ({ticker})...")
        try:
            # Download data
            data = yf.download(ticker, start=start_date, end=end_date) 
            # Keep only relevant columns and rename them
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            print(data.columns)
            print(f"  Successfully downloaded {len(data)} rows")

            if data is not None:
                df = data
                
        except Exception as e:
                print(f"  Error downloading {ticker}: {str(e)}")
        
        # Initial data info
        print(type(df))
        print(f"\nInitial data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Step 1: Remove any rows where all data is NaN (shouldn't happen with concat, but just in case)
        df = df.dropna(how='all')
        
        # Forward fill for up to 5 days (for holidays, etc.)
        df = df.fillna(method='ffill', limit=5)
        
        # Backward fill for the beginning if first few days are NaN
        df = df.fillna(method='bfill', limit=5)
        
        # Check for and handle any remaining NaNs
        remaining_nulls = df.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"\nWarning: {remaining_nulls} NaN values remain after cleaning. Dropping these rows.")
            df = df.dropna()
    
    return df

def _save_data(df, filename='assets.csv'):
    """
    Save the cleaned data to CSV file
    """
    if df is not None:
        path = ROOT / 'data' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path)
        print(f"\nData saved to {path}")

if __name__ == "__main__":
    spy_constituents_df = pd.read_csv(URL)
    sp500_tickers = spy_constituents_df['symbol'].tolist()

    print("SPY benchmark downloading")
    spy = yf_assets_data(tickers=["SPY"], years=YEARS)
    if spy is not None:
        _save_data(spy, "benchmark.csv")

    print("SPY constituents downloading...")
    assets = yf_assets_data(tickers=sp500_tickers, years=YEARS)
    if assets is not None:
        _save_data(assets, "assets.csv")
        
    print("\nProcess completed successfully!")