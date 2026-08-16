"""Walk-forward volatility backtest on FRED daily data + regime-conditional scorecard.

    python examples/run_backtest.py --series sp500 --test-start 2022-01-03
Writes results/<series>_summary.md (tables) used by the README.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volregime.backtest import dm_table, var_table, walk_forward  # noqa: E402
from volregime.data import load, log_returns  # noqa: E402
from volregime.models import DEFAULT_MODELS  # noqa: E402
from volregime.regimes import cusum, hmm_regimes, robust_zscore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="sp500")
    ap.add_argument("--test-start", default="2022-01-03")
    ap.add_argument("--step", type=int, default=1)
    args = ap.parse_args()

    df = load()
    r = log_returns(df[args.series])
    start = int(np.searchsorted(r.index, pd.Timestamp(args.test_start)))
    if start < 300:
        raise SystemExit("need >=300 obs before test start")
    res = walk_forward(r, DEFAULT_MODELS, start=start, step=args.step)
    summ = res.summary()
    dm = dm_table(res, benchmark="EWMA0.94", loss="qlike")
    var = var_table(res, dof=5.0)

    reg = hmm_regimes(r, n_states=2, train_end=start)
    reg_test = reg.loc[res.forecasts.index]
    L = res.losses()
    rows = []
    for m in res.forecasts.columns:
        for s, label in [(0, "calm"), (1, "stressed")]:
            mask = reg_test == s
            rows.append({"model": m, "regime": label, "n": int(mask.sum()), "QLIKE": float(L[m]["qlike"][mask].mean())})
    regtab = pd.DataFrame(rows).pivot(index="model", columns="regime", values="QLIKE")

    z = robust_zscore(r).loc[res.forecasts.index]
    cs = cusum(r ** 2).loc[res.forecasts.index]

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    md = [f"# {args.series.upper()} — walk-forward results (test from {args.test_start}, n={len(res.forecasts)})", "",
          "## Loss summary (lower is better)", summ.round(4).to_markdown(), "",
          "## Diebold–Mariano vs EWMA(0.94), QLIKE (negative DM = better than benchmark)", dm.round(4).to_markdown(), "",
          "## Regime-conditional QLIKE (HMM fitted on training window only)", regtab.round(4).to_markdown(), "",
          "## VaR backtests (Student-t, dof=5)", var.round(4).to_markdown(), "",
          f"## Anomalies\n- robust-z (|z|>4, 63d MAD): {int(z['anomaly'].sum())} days\n- CUSUM variance-shift alarms: {int((cs['alarm']!=0).sum())}\n- days in stressed regime: {int((reg_test==1).sum())} / {len(reg_test)}"]
    (out / f"{args.series}_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
