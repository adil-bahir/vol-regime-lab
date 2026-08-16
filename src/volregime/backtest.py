"""Walk-forward evaluation with lookahead enforced by construction.

For each target day t in the test window, every model receives returns[:t] (strictly before t)
and emits a variance forecast for t. The realised proxy is r_t^2. Losses:

  MSE   : (r_t^2 - h_t)^2                       — consistent but spike-dominated
  QLIKE : log(h_t) + r_t^2 / h_t                — robust loss of Patton (2011)
  preferred

Diebold–Mariano (1995) tests with Newey–West HAC variance compare each model to a benchmark
on the loss differential
Harvey–Leybourne–Newbold small-sample correction applied.

VaR backtests (Kupiec 1995 unconditional coverage
Christoffersen 1998 independence) are run
on the 1% and 5% one-day VaR implied by each variance forecast under a Student-t with the
GARCH-fitted degrees of freedom (fallback: normal).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .models import Model


@dataclass
class BacktestResult:
    forecasts: pd.DataFrame     # columns = model names, index = test dates, values = variance forecast
    realised: pd.Series         # r_t^2
    returns: pd.Series          # r_t

    def losses(self) -> dict[str, pd.DataFrame]:
        out = {}
        for m in self.forecasts.columns:
            h = self.forecasts[m].clip(lower=1e-8)
            out[m] = pd.DataFrame({
                "mse": (self.realised - h) ** 2,
                "qlike": np.log(h) + self.realised / h,
            })
        return out

    def summary(self) -> pd.DataFrame:
        L = self.losses()
        rows = []
        for m, df in L.items():
            rows.append({"model": m, "MSE": df["mse"].mean(), "QLIKE": df["qlike"].mean(),
                         "RMSE_vol": float(np.sqrt(np.mean((np.sqrt(self.realised) - np.sqrt(self.forecasts[m])) ** 2)))})
        return pd.DataFrame(rows).set_index("model").sort_values("QLIKE")


def walk_forward(returns: pd.Series, models: list[Model], start: int, end: int | None = None, step: int = 1,
                 floor_frac: float = 0.05) -> BacktestResult:
    """Run models on expanding window. `start` = index of first target day.
    Every forecast is floored at `floor_frac` x trailing-252d variance so QLIKE cannot be dominated by a
    single degenerate near-zero forecast (applied identically to all models)."""
    end = end or len(returns)
    idx = list(range(start, end, step))
    fc = {m.name: [] for m in models}
    for t in idx:
        hist = returns.iloc[:t]                    # strictly before t
        floor = floor_frac * float(hist.iloc[-252:].var(ddof=0))   # guard against degenerate near-zero forecasts
        for m in models:
            fc[m.name].append(max(m.forecast(hist), floor))
    dates = returns.index[idx]
    forecasts = pd.DataFrame(fc, index=dates)
    realised = (returns.iloc[idx] ** 2).rename("rv")
    return BacktestResult(forecasts, realised, returns.iloc[idx])


# ----------------------------------------------------------------------------- Diebold–Mariano
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM statistic (HLN-corrected) and two-sided p-value. Negative => model A has lower loss."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    dbar = d.mean()
    # Newey–West long-run variance with lag h-1 (Bartlett)
    lag = max(h - 1, 0)
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, lag + 1):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2 * (1 - k / (lag + 1)) * gk
    dm = dbar / np.sqrt(lrv / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(hln), df=n - 1))
    return float(hln), float(p)


def dm_table(result: BacktestResult, benchmark: str, loss: str = "qlike") -> pd.DataFrame:
    L = result.losses()
    rows = []
    for m in result.forecasts.columns:
        if m == benchmark:
            continue
        stat, p = diebold_mariano(L[m][loss].to_numpy(), L[benchmark][loss].to_numpy())
        rows.append({"model": m, "vs": benchmark, "loss": loss, "DM": stat, "p_value": p,
                     "better": "yes" if (stat < 0 and p < 0.05) else ("worse" if (stat > 0 and p < 0.05) else "n.s.")})
    return pd.DataFrame(rows).set_index("model")


# ----------------------------------------------------------------------------- VaR backtests
def var_exceedances(returns: pd.Series, var_forecast: pd.Series, alpha: float, dof: float | None = None) -> pd.Series:
    q = stats.t.ppf(alpha, dof) * np.sqrt((dof - 2) / dof) if dof and dof > 2 else stats.norm.ppf(alpha)
    var = q * np.sqrt(var_forecast)
    return (returns < var).astype(int)


def kupiec(hits: pd.Series, alpha: float) -> tuple[float, float]:
    n, x = len(hits), int(hits.sum())
    p_hat = x / n if n else 0.0
    if x == 0 or x == n:
        lr = -2 * (n * np.log(1 - alpha) if x == 0 else n * np.log(alpha))
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha) - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits: pd.Series) -> tuple[float, float]:
    h = hits.to_numpy()
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    p01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(a, b, pa, pb):
        return (a * np.log(1 - pa) if a and pa < 1 else 0) + (b * np.log(pa) if b and pa > 0 else 0)
    l0 = ll(n00 + n10, n01 + n11, p, p) if 0 < p < 1 else 0.0
    l1 = ll(n00, n01, p01, p01) + ll(n10, n11, p11, p11)
    lr = -2 * (l0 - l1)
    return float(lr), float(1 - stats.chi2.cdf(max(lr, 0), 1))


def var_table(result: BacktestResult, alphas=(0.01, 0.05), dof: float | None = None) -> pd.DataFrame:
    rows = []
    for m in result.forecasts.columns:
        for a in alphas:
            hits = var_exceedances(result.returns, result.forecasts[m], a, dof)
            lr_uc, p_uc = kupiec(hits, a)
            lr_ind, p_ind = christoffersen(hits)
            rows.append({"model": m, "alpha": a, "hit_rate": hits.mean(), "kupiec_p": p_uc, "christoffersen_p": p_ind})
    return pd.DataFrame(rows).set_index(["model", "alpha"])
