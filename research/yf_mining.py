import yfinance as yf
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
HISTORICAL_COMPONENTS = ROOT / 'data' / 'sp_500_historical_components.csv'
UNIVERSE_OUT          = ROOT / 'data' / 'assets_universe_tickers.csv'
ASSETS_OUT             = ROOT / 'data' / 'assets_universe_data.csv'
SPY_OUT               = ROOT / 'data' / 'spy.csv'


def build_universe(components_path: Path) -> tuple[list[str], str, str]:
    """
    Parse sp_500_historical_components.csv and return
    (sorted unique tickers, start_date string, end_date string).
    """
    df = pd.read_csv(components_path, parse_dates=['date'])
    all_tickers: set[str] = set()
    for row in df['tickers']:
        all_tickers.update(t.strip() for t in str(row).split(',') if t.strip())
    start = df['date'].min().strftime('%Y-%m-%d')
    end   = df['date'].max().strftime('%Y-%m-%d')
    return sorted(all_tickers), start, end


def download_ohlcv(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Batch-download adjusted close and volume for all tickers.
    Returns (close_df, volume_df); tickers with no data are dropped.
    """
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # yfinance returns MultiIndex columns for multiple tickers, flat for one
    if isinstance(raw.columns, pd.MultiIndex):
        close  = raw['Close']
        volume = raw['Volume']
    else:
        close  = raw[['Close']].rename(columns={'Close': tickers[0]})
        volume = raw[['Volume']].rename(columns={'Volume': tickers[0]})

    close  = close.dropna(how='all').dropna(axis=1, how='all')
    volume = volume.dropna(how='all').dropna(axis=1, how='all')

    # Align both to the same tickers/dates
    common_tickers = close.columns.intersection(volume.columns)
    close  = close[common_tickers]
    volume = volume[common_tickers]

    print(f"  Final matrix: {close.shape[0]} days x {close.shape[1]} tickers")
    return close, volume


if __name__ == "__main__":
    # 1. Build universe from historical components
    print("Reading historical S&P 500 components...")
    universe, start_date, end_date = build_universe(HISTORICAL_COMPONENTS)
    print(f"  {len(universe)} unique tickers found")
    print(f"  Date range in file: {start_date} to {end_date}")

    pd.DataFrame({'symbol': universe}).to_csv(UNIVERSE_OUT, index=False)
    print(f"  Universe saved to {UNIVERSE_OUT}")

    # 2. Download close + volume for all constituents
    close, volume = download_ohlcv(universe, start_date, end_date)
    close.index.name  = 'Date'
    volume.index.name = 'Date'
    assets = pd.merge(close, volume)
    assets.to_csv(ASSETS_OUT)
    print(f"  Assets  saved to {ASSETS_OUT}")

    # 3. Download SPY close + volume over the same period
    print(f"\nDownloading SPY from {start_date} to {end_date}...")
    spy_raw = yf.download('SPY', start=start_date, end=end_date,
                          auto_adjust=True, progress=False)
    spy = spy_raw[['Close', 'Volume']]
    spy.index.name = 'Date'
    spy.to_csv(SPY_OUT)
    print(f"  SPY saved to {SPY_OUT} ({len(spy)} rows)")

    print("\nDone.")
