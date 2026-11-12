"""
Label construction — one definition, shared by both dashboard pages and every
script.

Five private copies of this function existed, and all five carried the same
defect. The label is a forward return over `horizon` bars, so the last
`horizon` rows have no observable outcome: `Future_Return` is NaN there. Each
copy then did

    out["Target"] = (out["Future_Return"] > threshold).astype(int)
    return out.dropna(subset=["Target"])

intending the dropna to remove those censored rows. It removed nothing. A NaN
compared against a number is False, not NaN, and False casts to 0 — so the
censored tail was labelled "did not rise" and kept. In a market that rose every
single day, the last `horizon` rows of every training window were negatives.

That tail is the most recent data in each window, sitting directly against the
test fold, and the walk-forward embargo exists precisely because those rows'
outcomes are unobservable at training time. Fabricating a 0 for them is a
label-integrity error of the same family the embargo guards against, in every
script and both pages, on every fold.

Censored rows are dropped here before the label is formed, so a Target only
exists where the future return does.
"""
from __future__ import annotations

import pandas as pd


def add_target(df: pd.DataFrame, horizon: int, threshold_pct: float) -> pd.DataFrame:
    """Label each row 1 if Close rises more than `threshold_pct` % over `horizon` bars.

    Rows whose forward window runs past the end of the frame have no outcome
    and are dropped, not labelled. `Future_Return` is kept for signal-quality
    reporting.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1 bar, got {horizon}")
    out = df.copy()
    out["Future_Return"] = out["Close"].pct_change(horizon).shift(-horizon)
    # Censored rows: no observable outcome. Drop before the comparison, because
    # a NaN compared against the threshold is False and would become a 0.
    out = out.dropna(subset=["Future_Return"])
    out["Target"] = (out["Future_Return"] > threshold_pct / 100.0).astype(int)
    return out.reset_index(drop=True)
