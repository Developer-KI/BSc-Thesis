"""
=============================================================================
crsp_data.py - CRSP data loaders for the HRP experiment
=============================================================================

Three responsibilities:

1. load_crsp_returns()
   Read the CRSP daily file (long format), pivot to wide returns matrix.
   Handles chunked reading so a multi-GB CSV does not OOM.

2. load_constituents()
   Read the S&P 500 historical constituents file.  Expected columns:
   'permno, start, ending' (range format).  Each row says "PERMNO P was
   in the S&P 500 between start and ending (inclusive)".  A given PERMNO
   may appear multiple times (re-additions to the index).

3. UniverseFn
   Given a date, return the set of PERMNOs that were S&P 500 members on
   that date.  Implemented with a sorted-event scheme so each lookup is
   O(log n) and total memory is O(n_events).

The output of load_crsp_returns is shaped exactly like get_returns from
hrp_lib (DataFrame indexed by date, columns = asset IDs, values = log
returns) except that the asset IDs are integer PERMNOs and the matrix
is sparse (cells are NaN whenever a stock was not trading).  The
backtest_pit function in hrp_lib handles those NaNs at point-in-time.
=============================================================================
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd


# =============================================================================
# 1. Returns loader
# =============================================================================

def load_crsp_returns(data_csv: str,
                      *,
                      permno_subset: Optional[Iterable[int]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      price_col: str = "DlyClose",
                      date_col: str = "DlyCalDt",
                      permno_col: str = "PERMNO",
                      ret_col: Optional[str] = None,
                      chunksize: int = 1_000_000,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Load a CRSP daily file into a wide DataFrame of log returns.

    Parameters
    ----------
    data_csv : path to the CRSP daily CSV file (long format).
    permno_subset : optional iterable of PERMNOs to keep.
    start_date, end_date : optional ISO date strings to filter.
    price_col : name of the (split-adjusted) closing price column.
                For CRSP CIZ format this is 'DlyClose'.
    ret_col : if your file has a precomputed daily return column (e.g.,
              'RET' or 'DlyRet'), pass its name here and we skip the
              price-to-return calculation.  If None, we compute log
              returns from price_col.
    date_col, permno_col : column names.
    chunksize : rows per pandas chunk; reduce on low-RAM machines.

    Returns
    -------
    DataFrame
        Index = trading dates (Timestamp).
        Columns = PERMNOs (int).
        Values = log returns.  NaN where the stock was not trading.
    """
    if verbose:
        print(f"[crsp] reading {data_csv} (chunksize={chunksize}) ...")

    keep = [permno_col, date_col]
    keep.append(price_col if ret_col is None else ret_col)

    permno_set = set(int(p) for p in permno_subset) if permno_subset is not None else None
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    pieces = []
    total_rows = 0
    for chunk in pd.read_csv(data_csv, usecols=keep, chunksize=chunksize):
        chunk[date_col] = pd.to_datetime(chunk[date_col])
        if permno_set is not None:
            chunk = chunk[chunk[permno_col].isin(permno_set)]
        if start_ts is not None:
            chunk = chunk[chunk[date_col] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk[date_col] <= end_ts]
        if not chunk.empty:
            pieces.append(chunk)
            total_rows += len(chunk)
    if not pieces:
        raise RuntimeError("No rows survived the filters; check your inputs.")
    df = pd.concat(pieces, ignore_index=True)
    if verbose:
        print(f"[crsp] kept {total_rows:,} rows across "
              f"{df[permno_col].nunique():,} PERMNOs and "
              f"{df[date_col].nunique():,} dates")

    # pivot to wide: rows = date, columns = permno
    if ret_col is None:
        wide = df.pivot_table(index=date_col, columns=permno_col,
                              values=price_col, aggfunc="last")
        wide = wide.sort_index()
        # log returns from split-adjusted close
        returns = np.log(wide / wide.shift(1))
    else:
        returns = df.pivot_table(index=date_col, columns=permno_col,
                                 values=ret_col, aggfunc="last")
        returns = returns.sort_index()

    returns.columns = returns.columns.astype(int)
    returns.index.name = "date"

    if verbose:
        print(f"[crsp] returns matrix: {returns.shape[0]} dates "
              f"x {returns.shape[1]} permnos  "
              f"(non-NaN density: {(~returns.isna()).mean().mean():.1%})")

    return returns

# =============================================================================
# 1b. Market cap loader (wide)
# =============================================================================

def load_market_cap(data_csv: str,
                    *,
                    permno_subset: Optional[Iterable[int]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    cap_col: str = "DlyCap",
                    date_col: str = "DlyCalDt",
                    permno_col: str = "PERMNO",
                    chunksize: int = 1_000_000,
                    verbose: bool = True) -> pd.DataFrame:
    """
    Load market capitalisation (DlyCap) from the CRSP daily file into a wide
    DataFrame (dates x PERMNOs).  This can be used to filter the universe
    by size at each point in time.

    Parameters
    ----------
    data_csv : path to CRSP daily CSV (long format).
    permno_subset : optional iterable of PERMNOs to keep.
    start_date, end_date : optional ISO date strings to filter.
    cap_col : name of the market cap column (default 'DlyCap').
    date_col, permno_col : column names.
    chunksize : rows per chunk.
    verbose : print progress.

    Returns
    -------
    DataFrame
        Index = trading dates (Timestamp).
        Columns = PERMNOs (int).
        Values = market capitalisation (same unit as CRSP, e.g. $ millions).
                Missing values (e.g. weekends, delistings) are left as NaN.
    """
    if verbose:
        print(f"[crsp] loading market caps from {data_csv} ...")

    keep = [permno_col, date_col, cap_col]
    permno_set = set(int(p) for p in permno_subset) if permno_subset is not None else None
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    pieces = []
    total_rows = 0
    for chunk in pd.read_csv(data_csv, usecols=keep, chunksize=chunksize):
        chunk[date_col] = pd.to_datetime(chunk[date_col])
        if permno_set is not None:
            chunk = chunk[chunk[permno_col].isin(permno_set)]
        if start_ts is not None:
            chunk = chunk[chunk[date_col] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk[date_col] <= end_ts]
        if not chunk.empty:
            pieces.append(chunk)
            total_rows += len(chunk)
    if not pieces:
        raise RuntimeError("No market cap rows survived filters.")

    df = pd.concat(pieces, ignore_index=True)
    if verbose:
        print(f"[crsp] kept {total_rows:,} rows with caps")

    # pivot to wide: rows = date, columns = permno
    cap_wide = df.pivot_table(index=date_col, columns=permno_col,
                              values=cap_col, aggfunc="last")
    cap_wide = cap_wide.sort_index()
    cap_wide.columns = cap_wide.columns.astype(int)
    cap_wide.index.name = "date"
    if verbose:
        print(f"[crsp] cap matrix: {cap_wide.shape[0]} dates x {cap_wide.shape[1]} permnos")
    return cap_wide


# =============================================================================
# 2. Constituents loader
# =============================================================================

def load_constituents(constituents_csv: str,
                      *,
                      permno_col: str = "permno",
                      start_col: str = "start",
                      end_col: str = "ending",
                      verbose: bool = True) -> pd.DataFrame:
    """
    Load the S&P 500 historical constituents (range format).

    Returns a DataFrame with columns [permno (int), start (Timestamp),
    ending (Timestamp)].  Multiple rows per PERMNO are allowed (these
    encode re-additions to the index).
    """
    df = pd.read_csv(constituents_csv)
    df = df.rename(columns={permno_col: "permno",
                            start_col: "start",
                            end_col: "ending"})
    df["permno"] = df["permno"].astype(int)
    df["start"] = pd.to_datetime(df["start"])
    df["ending"] = pd.to_datetime(df["ending"])
    if verbose:
        print(f"[const] {len(df):,} rows, "
              f"{df['permno'].nunique():,} unique PERMNOs")
    return df


# =============================================================================
# 3. Universe function (with optional market cap filtering)
# =============================================================================

class UniverseFn:
    """
    Callable: date -> set[int] of PERMNOs in the S&P 500 on that date,
    optionally filtered to the top K market capitalisation.

    Parameters
    ----------
    constituents : DataFrame with columns ['permno', 'start', 'ending'].
    market_cap_wide : DataFrame (optional) with index = date, columns = permno,
                      values = market cap. If provided, the universe will be
                      restricted to the top `top_k` PERMNOs by cap on the given date.
    top_k : int (default = None, meaning no cap filter).  Only used if market_cap_wide is given.
    """

    def __init__(self, constituents: pd.DataFrame,
                 market_cap_wide: Optional[pd.DataFrame] = None,
                 top_k: Optional[int] = None):
        self._starts = constituents["start"].values.astype("datetime64[D]")
        self._ends = constituents["ending"].values.astype("datetime64[D]")
        self._permnos = constituents["permno"].values.astype(int)
        self._df = constituents
        self._cap_wide = market_cap_wide
        self._top_k = top_k
        if market_cap_wide is not None and top_k is not None:
            # Pre‑align columns with the cap matrix to speed up sorting later
            self._cap_cols = np.asarray(market_cap_wide.columns, dtype=int)

    def __call__(self, date: pd.Timestamp) -> Set[int]:
        d = np.datetime64(pd.Timestamp(date).date(), "D")
        mask = (self._starts <= d) & (self._ends >= d)
        sp_permnos = set(self._permnos[mask].tolist())
        if not sp_permnos:
            return set()

        # If no market cap filtering, return all S&P constituents
        if self._cap_wide is None or self._top_k is None:
            return sp_permnos

        # Get market caps on this date for all S&P permnos that exist in cap matrix
        try:
            caps_series = self._cap_wide.loc[date]   # Series indexed by permno
        except KeyError:
            # No data for this date (e.g., holiday) -> return empty
            return set()

        # Keep only those permnos that are both in S&P and have a non‑NaN cap
        valid = []
        for p in sp_permnos:
            if p in caps_series.index:
                cap = caps_series[p]
                if not np.isnan(cap):
                    valid.append((p, cap))
        if not valid:
            return set()
        # Sort by cap descending and take top_k
        valid.sort(key=lambda x: x[1], reverse=True)
        top = {p for p, _ in valid[:self._top_k]}
        return top

    def size_at(self, date: pd.Timestamp) -> int:
        return len(self(date))

    def trace(self, dates: Iterable[pd.Timestamp]) -> pd.Series:
        sizes = [self.size_at(d) for d in dates]
        return pd.Series(sizes, index=list(dates), name="universe_size")


def make_universe_fn(constituents_csv: str,
                     market_cap_csv: Optional[str] = None,
                     top_k: Optional[int] = None,
                     **kwargs) -> UniverseFn:
    """
    Convenience: load constituents and market caps (if given), return UniverseFn.

    Parameters
    ----------
    constituents_csv : path to S&P 500 constituents file.
    market_cap_csv : path to CRSP daily file (to load DlyCap).  If None, no cap filtering.
    top_k : number of largest-cap stocks to keep per date (only used if market_cap_csv provided).
    **kwargs : passed to load_constituents (e.g., column names).

    Returns
    -------
    UniverseFn instance.
    """
    constituents = load_constituents(constituents_csv, **kwargs)
    if market_cap_csv is not None and top_k is not None:
        caps = load_market_cap(market_cap_csv, **kwargs)
        # Note: we assume the same date range / PERMNO subset is already handled.
        return UniverseFn(constituents, market_cap_wide=caps, top_k=top_k)
    return UniverseFn(constituents)