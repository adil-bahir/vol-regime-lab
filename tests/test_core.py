import numpy as np
import pandas as pd
import pytest

from volregime.backtest import christoffersen, diebold_mariano, kupiec, walk_forward
from volregime.data import load, log_returns
from volregime.models import EWMA, GARCH11, HARRV, HARRVol, HistoricalVariance
from volregime.regimes import cusum, hmm_regimes, robust_zscore


@pytest.fixture(scope="module")
def r():
    return log_returns(load()["sp500"])


def test_no_lookahead_by_construction(r):
    """A model that returns the *next* return squared would be a perfect oracle if the driver leaked;
    the driver only ever passes returns[:t], so an oracle cannot exist. We check the contract directly."""
    seen = []

    class Spy(EWMA):
        name = "spy"
        def forecast(self, hist):
            seen.append(hist.index[-1])
            return 1.0
    res = walk_forward(r, [Spy()], start=len(r) - 5)
    for hist_last, target in zip(seen, res.forecasts.index):
        assert hist_last < target


def test_models_return_positive_finite(r):
    hist = r.iloc[:800]
    for m in [HistoricalVariance(), EWMA(), GARCH11(), HARRV(), HARRVol()]:
        v = m.forecast(hist)
        assert np.isfinite(v) and v > 0, m.name


def test_dm_symmetric_and_null():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    s1, p1 = diebold_mariano(a, b)
    s2, p2 = diebold_mariano(b, a)
    assert abs(s1 + s2) < 1e-12 and p1 == p2
    assert p1 > 0.01           # same distribution -> should not reject at 1% (probabilistic but stable with seed)


def test_kupiec_and_christoffersen_calibrated():
    rng = np.random.default_rng(1)
    hits = pd.Series((rng.random(2000) < 0.05).astype(int))
    _, p = kupiec(hits, 0.05)
    assert p > 0.05
    _, p_ind = christoffersen(hits)
    assert p_ind > 0.05
    bad = pd.Series((rng.random(2000) < 0.15).astype(int))
    _, p_bad = kupiec(bad, 0.05)
    assert p_bad < 0.001


def test_walk_forward_shapes_and_losses(r):
    res = walk_forward(r, [EWMA(), HistoricalVariance()], start=len(r) - 60, step=1)
    assert res.forecasts.shape == (60, 2)
    s = res.summary()
    assert set(s.columns) == {"MSE", "QLIKE", "RMSE_vol"} and np.isfinite(s.to_numpy()).all()


def test_regimes_and_anomalies(r):
    reg = hmm_regimes(r, n_states=2, train_end=len(r) - 300)
    assert set(reg.unique()) <= {0, 1}
    assert r[reg == 1].abs().mean() > r[reg == 0].abs().mean()   # state 1 is the high-vol state
    z = robust_zscore(r)
    assert z["anomaly"].sum() > 0
    cs = cusum(r ** 2)
    assert cs["alarm"].abs().sum() > 0
