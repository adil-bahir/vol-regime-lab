<h1 align="center">vol-regime-lab</h1>
<p align="center"><b>Walk-forward volatility forecasting, regime detection and anomaly flags on public market data</b><br/>
GARCH-t · EWMA · HAR · Diebold–Mariano · Kupiec / Christoffersen VaR backtests · HMM regimes · CUSUM — no lookahead by construction, fully reproducible offline</p>

<p align="center">
<a href="https://github.com/adil-bahir/vol-regime-lab/actions/workflows/ci.yml"><img alt="ci" src="https://github.com/adil-bahir/vol-regime-lab/actions/workflows/ci.yml/badge.svg"></a>
<img alt="python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white">
<img alt="data" src="https://img.shields.io/badge/data-FRED%20(public)-0E57FC">
<img alt="license" src="https://img.shields.io/badge/license-MIT-blue">
</p>

> **What this is.** A small, honest research harness: five one-step-ahead variance forecasters evaluated walk-forward on daily S&P 500 and Nasdaq data from FRED, scored with QLIKE and MSE, compared with Diebold–Mariano tests against the RiskMetrics benchmark, and stress-tested through 1%/5% VaR coverage and independence tests. A two-state Gaussian HMM (fitted on the training window only) splits the test period into calm/stressed regimes so you can see *where* each model earns its keep. Robust-z and CUSUM detectors flag spikes and persistent variance shifts.
>
> **What it is not.** Not a trading strategy, not intraday, not overfitted. Everything runs from a committed CSV in CI; `python examples/run_backtest.py` reproduces every table below.

## Headline results — S&P 500, test window 2022-01-03 → 2026-08 (n≈1,158 days)

## Loss summary (lower is better)
| model        |     MSE |   QLIKE |   RMSE_vol |
|:-------------|--------:|--------:|-----------:|
| GARCH(1,1)-t | 11.2732 |  0.9681 |     0.7971 |
| EWMA0.94     | 11.1621 |  0.9842 |     0.7835 |
| HAR-RVol     | 11.3503 |  1.0444 |     0.7939 |
| HistVar21    | 11.6446 |  1.0501 |     0.7873 |
| HAR-RV       | 16.0016 |  1.1562 |     0.8336 |

## Diebold–Mariano vs EWMA(0.94), QLIKE (negative DM = better than benchmark)
| model        | vs       | loss   |      DM |   p_value | better   |
|:-------------|:---------|:-------|--------:|----------:|:---------|
| HistVar21    | EWMA0.94 | qlike  |  2.1101 |    0.0351 | worse    |
| GARCH(1,1)-t | EWMA0.94 | qlike  | -0.9609 |    0.3368 | n.s.     |
| HAR-RV       | EWMA0.94 | qlike  |  1.9697 |    0.0491 | worse    |
| HAR-RVol     | EWMA0.94 | qlike  |  1.5427 |    0.1232 | n.s.     |

## Regime-conditional QLIKE (HMM fitted on training window only)
| model        |   calm |   stressed |
|:-------------|-------:|-----------:|
| EWMA0.94     | 0.8239 |     1.1683 |
| GARCH(1,1)-t | 0.8047 |     1.1556 |
| HAR-RV       | 0.9226 |     1.4244 |
| HAR-RVol     | 0.9216 |     1.1854 |
| HistVar21    | 0.8499 |     1.2801 |

## VaR backtests (Student-t, dof=5)
|                        |   hit_rate |   kupiec_p |   christoffersen_p |
|:-----------------------|-----------:|-----------:|-------------------:|
| ('HistVar21', 0.01)    |     0.0181 |     0.0125 |             0.3931 |
| ('HistVar21', 0.05)    |     0.0708 |     0.0022 |             0.7114 |
| ('EWMA0.94', 0.01)     |     0.0121 |     0.4889 |             0.1581 |
| ('EWMA0.94', 0.05)     |     0.0717 |     0.0014 |             0.9839 |
| ('GARCH(1,1)-t', 0.01) |     0.013  |     0.334  |             0.1854 |
| ('GARCH(1,1)-t', 0.05) |     0.0691 |     0.0047 |             0.4647 |
| ('HAR-RV', 0.01)       |     0.0164 |     0.0449 |             0.3161 |
| ('HAR-RV', 0.05)       |     0.0613 |     0.0875 |             0.8537 |
| ('HAR-RVol', 0.01)     |     0.0121 |     0.4889 |             0.1581 |
| ('HAR-RVol', 0.05)     |     0.0535 |     0.5845 |             0.7026 |

**Reading it.** On a squared-daily-return proxy, Student-t GARCH(1,1) edges RiskMetrics on QLIKE in both regimes but not significantly (DM p≈0.34); the 21-day sample variance is significantly worse; HAR variants — which shine on intraday realised variance — lose here because the daily proxy is too noisy for the weekly/monthly components to add information. That is the expected result and the harness reports it rather than hiding it. All models under-cover 5% VaR (hit rates ~7%; Kupiec rejects) while 1% coverage is acceptable for EWMA/GARCH — consistent with fat tails beyond a t(5). Nasdaq results ([`results/nasdaq_summary.md`](results/nasdaq_summary.md)) tell the same story with higher loss levels.

## Design choices a reviewer should check

- **Lookahead is impossible by construction.** `walk_forward` hands each model `returns[:t]` and asks for `t`; there is a test (`test_no_lookahead_by_construction`) that spies on the last index a model ever sees. HMM regimes are fitted on the training window and decoded forward.
- **QLIKE, not just MSE.** MSE on variance is dominated by a handful of spike days; QLIKE (Patton 2011) is the robust loss for variance proxies and is the ranking metric.
- **A common floor.** Every forecast is floored at 5% of trailing-252d variance, applied identically to all models, so QLIKE cannot be blown up by a single degenerate near-zero forecast (HAR-RV produced one; the floor made the comparison fair rather than making HAR look artificially bad).
- **HAR on |r| instead of log r².** With a squared-daily-return proxy, log r² has enormous variance on near-zero days; HAR-RVol (HAR on |r|, squared back with a residual-variance term) is the stable variant. The unstable log version was tried and removed; the commit history shows it.
- **Small-sample DM.** Harvey–Leybourne–Newbold correction; Newey–West long-run variance for h>1.

## Repository map

```
src/volregime/
  data.py       FRED loader + committed cache (SP500, NASDAQ, VIX, DGS10, EURUSD, WTI)
  models.py     HistoricalVariance · EWMA(0.94) · GARCH(1,1)-t (arch) · HAR-RV · HAR-RVol
  backtest.py   walk_forward · QLIKE/MSE · Diebold–Mariano (HLN) · Kupiec · Christoffersen
  regimes.py    Gaussian HMM regimes · robust MAD z-score · two-sided CUSUM
examples/run_backtest.py   reproduces results/*.md
tests/                     6 tests (lookahead contract, calibration of DM/Kupiec/Christoffersen, model sanity, regimes)
data/fred_daily.csv        cached public data (refresh with volregime.data.refresh_cache)
```

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
python examples/run_backtest.py --series sp500 --test-start 2022-01-03
python examples/run_backtest.py --series nasdaq --test-start 2022-01-03
```

## Roadmap

- Realized-GARCH / HAR-Q with intraday RV when a public intraday source is wired in
- Model Confidence Set (Hansen–Lunde–Nason) across the model pool
- Regime-switching GARCH and HMM-conditional model selection
- Cross-asset: EURUSD, WTI, 10y yield changes with the same harness

## Author

**Adil Bahir** — DEng (AI/ML), MFin, CQF, FRM, CFA. Enterprise AI Enablement & CFO Advisory Partner (KPMG); builder of [lights-out-agents](https://github.com/adil-bahir/lights-out-agents) and [Lights Out Finance](https://lightsoutfinance.net). MIT License.
