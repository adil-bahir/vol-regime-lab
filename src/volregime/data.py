"""Public market data via FRED, with a committed CSV cache so everything runs offline in CI.

Series (all daily, FRED ids):
  SP500 (S&P 500), NASDAQCOM (Nasdaq Composite), VIXCLS (CBOE VIX), DGS10 (10y UST yield),
  DEXUSEU (EUR/USD), DCOILWTICO (WTI crude).

Only the closing level is available from FRED, so realized variance here is the classic
squared-daily-return proxy (Andersen & Bollerslev, 1998), not intraday RV. HAR-RV is applied
to that proxy — the model structure is unchanged, the noise floor is higher. All results in
this repo are therefore conservative relative to what intraday data would give.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = {"SP500": "sp500", "NASDAQCOM": "nasdaq", "VIXCLS": "vix", "DGS10": "dgs10", "DEXUSEU": "eurusd", "DCOILWTICO": "wti"}
CACHE = Path(__file__).resolve().parents[2] / "data" / "fred_daily.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_fred(sid: str, timeout: int = 30) -> pd.Series:
    raw = urllib.request.urlopen(FRED.format(sid=sid), timeout=timeout).read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", sid]
    s = pd.to_numeric(df[sid], errors="coerce")
    s.index = pd.to_datetime(df["date"])
    return s.rename(SERIES.get(sid, sid))


def refresh_cache(path: Path = CACHE) -> pd.DataFrame:
    frames = [fetch_fred(sid) for sid in SERIES]
    df = pd.concat(frames, axis=1, sort=True)
    df.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load(path: Path = CACHE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
    return df


def log_returns(px: pd.Series) -> pd.Series:
    """Daily log returns in percent, NaNs (holidays) dropped."""
    px = px.dropna()
    return (100.0 * np.log(px).diff()).dropna().rename(f"{px.name}_ret")
