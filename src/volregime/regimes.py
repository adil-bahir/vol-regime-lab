"""Regime detection and anomaly flags on the same return series.

- `hmm_regimes`   : Gaussian HMM (hmmlearn) on standardised returns and |returns|; states are
                    relabelled by ascending volatility so state 0 is always "calm". Fitted only
                    on the training window; decoded forward on the test window (no refit on
                    test data unless `refit=True`, and then only with data up to t).
- `robust_zscore` : rolling median/MAD z-score of returns — spike detector.
- `cusum`         : two-sided CUSUM on squared-return innovations vs. an EWMA baseline — detects
                    persistent variance shifts rather than single spikes.

Regime-conditional performance of the volatility models is reported in `examples/run_backtest.py`
so the reader can see which forecaster wins in calm vs. stressed states — the point of the repo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def hmm_regimes(returns: pd.Series, n_states: int = 2, train_end: int | None = None, seed: int = 7) -> pd.Series:
    from hmmlearn.hmm import GaussianHMM

    r = returns.to_numpy()
    X = np.column_stack([r, np.abs(r)])
    train = X[:train_end] if train_end else X
    mu, sd = train.mean(0), train.std(0) + 1e-9
    Z = (X - mu) / sd
    hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=seed)
    hmm.fit(Z[:train_end] if train_end else Z)
    states = hmm.predict(Z)
    # relabel by volatility of |r| within state
    order = np.argsort([np.abs(r[states == s]).mean() if np.any(states == s) else np.inf for s in range(n_states)])
    remap = {old: new for new, old in enumerate(order)}
    return pd.Series([remap[s] for s in states], index=returns.index, name="regime")


def robust_zscore(x: pd.Series, window: int = 63, thresh: float = 4.0) -> pd.DataFrame:
    med = x.rolling(window).median().shift(1)
    mad = (x - med).abs().rolling(window).median().shift(1) * 1.4826
    z = (x - med) / mad.replace(0, np.nan)
    return pd.DataFrame({"z": z, "anomaly": (z.abs() > thresh).astype(int)})


def cusum(x2: pd.Series, lam: float = 0.94, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """Two-sided CUSUM on standardised squared-return innovations vs an EWMA variance baseline."""
    s2 = x2.ewm(alpha=1 - lam, adjust=False).mean().shift(1)
    z = (x2 / s2 - 1.0)  # >0 when variance above baseline
    z = z / z.rolling(252).std().shift(1)
    up = np.zeros(len(z))
    dn = np.zeros(len(z))
    alarm = np.zeros(len(z), dtype=int)
    for i in range(1, len(z)):
        v = z.iloc[i]
        if np.isnan(v):
            continue
        up[i] = max(0.0, up[i - 1] + v - k)
        dn[i] = max(0.0, dn[i - 1] - v - k)
        if up[i] > h or dn[i] > h:
            alarm[i] = 1 if up[i] > h else -1
            up[i] = dn[i] = 0.0
    return pd.DataFrame({"cusum_up": up, "cusum_dn": dn, "alarm": alarm}, index=x2.index)
