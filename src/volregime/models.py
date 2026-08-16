"""Volatility forecasters with a common one-step-ahead interface.

Every model implements `forecast(returns_up_to_t) -> variance forecast for t+1` and is only
ever fed data strictly before the target date by the walk-forward driver in `backtest.py`.
No model sees the future; the driver enforces it, not the model.

Models
------
- HistoricalVariance : rolling sample variance (naive benchmark)
- EWMA               : RiskMetrics, lambda=0.94 (industry benchmark)
- GARCH11            : Student-t GARCH(1,1) via `arch`, refit every `refit_every` steps
- HARRV              : Corsi (2009) heterogeneous autoregression on daily/weekly/monthly RV proxy
- HARRVol            : HAR on |r| (sqrt of RV proxy), squared back with residual-variance bias term
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class Model:
    name: str = "base"

    def forecast(self, r: pd.Series) -> float:  # variance forecast for next day, in %^2
        raise NotImplementedError


@dataclass
class HistoricalVariance(Model):
    window: int = 21
    name: str = "HistVar21"

    def forecast(self, r: pd.Series) -> float:
        return float(r.iloc[-self.window:].var(ddof=0))


@dataclass
class EWMA(Model):
    lam: float = 0.94
    name: str = "EWMA0.94"

    def forecast(self, r: pd.Series) -> float:
        # recursive: s2_t = lam*s2_{t-1} + (1-lam)*r_{t-1}^2 ; init with sample var of first 21 obs
        x = r.to_numpy()
        s2 = float(np.var(x[:21])) if len(x) >= 21 else float(np.var(x))
        for v in x[21:]:
            s2 = self.lam * s2 + (1 - self.lam) * v * v
        return s2


@dataclass
class GARCH11(Model):
    refit_every: int = 21
    dist: str = "t"
    name: str = "GARCH(1,1)-t"
    _fit: object = field(default=None, repr=False)
    _n_at_fit: int = field(default=-1, repr=False)

    def forecast(self, r: pd.Series) -> float:
        from arch import arch_model

        n = len(r)
        if self._fit is None or n - self._n_at_fit >= self.refit_every:
            am = arch_model(r, vol="GARCH", p=1, q=1, dist=self.dist, mean="Constant", rescale=False)
            self._fit = am.fit(disp="off", show_warning=False)
            self._n_at_fit = n
            self._params = self._fit.params
        p = self._params
        omega, alpha, beta, mu = p["omega"], p["alpha[1]"], p["beta[1]"], p["mu"]
        # filter conditional variance forward with fixed params on the full sample (no lookahead: r ends at t)
        eps = (r - mu).to_numpy()
        s2 = float(np.var(eps[:21]))
        for e in eps:
            s2 = omega + alpha * e * e + beta * s2
        return float(s2)


def _har_features(rv: pd.Series) -> pd.DataFrame:
    d = rv
    w = rv.rolling(5).mean()
    m = rv.rolling(22).mean()
    return pd.DataFrame({"d": d, "w": w, "m": m})


@dataclass
class HARRV(Model):
    window: int = 500
    name: str = "HAR-RV"

    def forecast(self, r: pd.Series) -> float:
        rv = (r ** 2).rename("rv")
        X = _har_features(rv).shift(1)  # features at t-1 predict rv at t
        df = pd.concat([rv, X], axis=1).dropna().iloc[-self.window:]
        A = np.column_stack([np.ones(len(df)), df[["d", "w", "m"]].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, df["rv"].to_numpy(), rcond=None)
        last = _har_features(rv).iloc[-1].to_numpy()          # features at t predict t+1
        return float(max(beta[0] + last @ beta[1:], 1e-8))


@dataclass
class HARRVol(Model):
    """HAR on the square root of the RV proxy (|r|), squared back with a residual-variance bias term.
    Far more stable than HAR on log r^2 when the proxy is squared daily returns (many near-zero days)."""
    window: int = 500
    name: str = "HAR-RVol"

    def forecast(self, r: pd.Series) -> float:
        v = r.abs().rename("v")
        X = _har_features(v).shift(1)
        df = pd.concat([v, X], axis=1).dropna().iloc[-self.window:]
        A = np.column_stack([np.ones(len(df)), df[["d", "w", "m"]].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, df["v"].to_numpy(), rcond=None)
        resid_var = float(np.mean((df["v"].to_numpy() - A @ beta) ** 2))
        last = _har_features(v).iloc[-1].to_numpy()
        mu = max(beta[0] + last @ beta[1:], 1e-4)
        return float(mu * mu + resid_var)


DEFAULT_MODELS = [HistoricalVariance(), EWMA(), GARCH11(), HARRV(), HARRVol()]
