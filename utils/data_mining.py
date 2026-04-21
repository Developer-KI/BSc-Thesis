# Data mining tools

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import warnings
import os
import glob

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
URL = "https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/sp500.csv"

START = "2015-01-02"   
MIN_YEARS = 5
YEARS = 10

DOWNLOAD = True
TOCSV = False


def yf_assets_data(tickers: List[str], years: float, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Download daily closing prices for a list of tickers.

    If start_date is provided (e.g., "2015-01-02"), the data window is:
        from start_date to start_date + years (forward window).
    If start_date is None or empty, the window is:
        from (today - years) to today (backward window).
    """
    end_date = datetime.now()
    
    if start_date:
        # Fixed forward window: start_date -> start_date + years
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = start + timedelta(days=years * 365)
        print(f"Using fixed start date: {start.strftime('%Y-%m-%d')}")
        print(f"Forward window ends at: {end.strftime('%Y-%m-%d')}")
    else:
        # Dynamic backward window: today - years -> today
        start = end_date - timedelta(days=years * 365)
        end = end_date
        print(f"Using dynamic window ending today")
    
    print(f"Downloading data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    print("-" * 50)

    close_series = {}

    for ticker in tqdm(tickers, desc="Downloading tickers", unit="ticker"):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
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


def convert_xlsx_to_csv(directory_path, delete_original=True):
    """
    Reads all xlsx files in a directory, converts them to CSV, and optionally deletes the original xlsx files.
    """
    xlsx_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    if not xlsx_files:
        print(f"No xlsx files found in {directory_path}")
        return []
    
    results = []
    
    for xlsx_file in xlsx_files:
        try:
            df = pd.read_excel(xlsx_file)
            csv_file = os.path.splitext(xlsx_file)[0] + ".csv"
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            if delete_original:
                os.remove(xlsx_file)
                status = "converted and original deleted"
            else:
                status = "converted (original kept)"
            
            results.append((xlsx_file, csv_file, status))
            print(f"✓ {os.path.basename(xlsx_file)} -> {os.path.basename(csv_file)} ({status})")
            
        except Exception as e:
            error_msg = f"Error processing {xlsx_file}: {str(e)}"
            results.append((xlsx_file, None, error_msg))
            print(f"✗ {error_msg}")
    
    return results


if __name__ == "__main__":
    if DOWNLOAD:
        spy_constituents_df = pd.read_csv(URL)
        csv_path = ROOT / 'data' / 'sp500_tickers.csv'
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        spy_constituents_df.to_csv(csv_path, index=False)
        print(f"SP500 tickers saved to {csv_path}")
        sp500_tickers = ["SPY"] + spy_constituents_df['symbol'].tolist()

        print("\nSPY and its constituents downloading...")
        # Pass START to the function; if START is None or empty string, use dynamic window
        start_arg = START if START and str(START).strip() else None
        assets = yf_assets_data(tickers=sp500_tickers, years=YEARS, start_date=start_arg)
        _save_data(assets, "assets.xlsx")

    if TOCSV:
        convert_xlsx_to_csv(ROOT / 'data', delete_original=False)

    print("\nProcess completed successfully!")