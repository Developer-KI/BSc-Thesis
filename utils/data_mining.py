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
MIN_YEARS = 5
ROOT = Path(__file__).resolve().parent.parent

def yf_assets_data(tickers: List[str], years: float) -> pd.DataFrame:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("-" * 50)

    close_series = {}

    for ticker in tqdm(tickers, desc="Downloading tickers", unit="ticker"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"  No data for {ticker}, skipping.")
                continue

            close = data['Close'].squeeze()
            close.name = ticker

            valid = close.dropna()
            span_days = (valid.index.max() - valid.index.min()).days if len(valid) > 0 else 0
            if span_days < MIN_YEARS * 365:
                print(f"  Skipping {ticker}: only {span_days / 365:.1f} years of data (need {MIN_YEARS}).")
                continue

            close_series[ticker] = close
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")

    if not close_series:
        raise ValueError("No data was downloaded.")

    df = pd.concat(close_series.values(), axis=1)
    df.index.name = "Date"

    df = df.dropna(how='all')
    df = df.ffill(limit=5)
    df = df.bfill(limit=5)

    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"\nWarning: {remaining_nulls} NaN values remain after cleaning. Dropping these rows.")
        df = df.dropna()

    print(f"\nFinal shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df

def _save_data(df: pd.DataFrame, filename: str = 'assets.xlsx'):
    path = ROOT / 'data' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.copy()
    df_out.index = df_out.index.strftime('%m/%d/%Y')
    df_out.to_excel(path)
    print(f"Data saved to {path}")

if __name__ == "__main__":
    spy_constituents_df = pd.read_csv(URL)
    csv_path = ROOT / 'data' / 'sp500_tickers.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    spy_constituents_df.to_csv(csv_path, index=False)
    print(f"SP500 tickers saved to {csv_path}")
    sp500_tickers = spy_constituents_df['symbol'].tolist()

    print("\nSPY constituents downloading...")
    assets = yf_assets_data(tickers=sp500_tickers, years=YEARS)
    _save_data(assets, "assets.xlsx")

    print("\nProcess completed successfully!")
