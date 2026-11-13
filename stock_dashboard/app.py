"""
Equity Signal Engine — landing page.

Streamlit multi-page app. Individual analyses live under pages/:
    1_Signal_Generator.py  — classifier training, calibration, in-sample signals
    2_Strategy_Backtest.py — walk-forward validation + event-driven backtest
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Equity Signal Engine",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Equity Signal Engine")
st.caption(
    "Walk-forward backtesting infrastructure for short-term equity direction "
    "prediction, built to answer a question honestly rather than to answer it well."
)

st.markdown("""
### The finding

Daily OHLCV technical indicators do not predict short-term direction on SPY.
That was the expected answer. The repository is the apparatus that establishes
it — embargoed folds, preprocessing fit per fold, an execution lag, intrabar
stops, confidence intervals that respect autocorrelation, and the whole search
grid published rather than its best cell.

**SPY, 10 years, 37 walk-forward folds, 2,183 out-of-sample days** (5-day holding,
0.5% threshold, 0.1% cost per leg; from the committed measurements in the README):

| | In-sample | Walk-forward (OOS) | Buy & hold |
|---|---:|---:|---:|
| Total return | 7.4% | **0.4%** | **237.9%** |
| Sharpe | 0.63 | **0.03** | 0.81 |

Stationary-bootstrap Sharpe 95% CI on OOS daily returns **[-0.55, 0.80]** · t-test p = **0.93**.
""")

_fig = Path(__file__).resolve().parents[1] / "docs" / "results_sensitivity.png"
if _fig.exists():
    st.image(
        str(_fig),
        caption="Out-of-sample Sharpe across 18 configurations of holding period, "
                "threshold and confidence cut. One cell clears p<0.05, which is "
                "what a grid of eighteen is expected to produce by chance.",
        use_container_width=True,
    )

st.markdown("""
### The pages

| Page | What it shows |
|------|---------------|
| **Strategy Backtest** | The walk-forward estimate first, then the in-sample fit it is being compared against, then performance by volatility regime |
| **Signal Generator** | The classifier itself: cross-validated metrics, calibration, feature importance, and the in-sample signals it produces |

Both pages default to two years so the walk-forward can run in seconds. Two
years is four folds and a handful of trades; the number that means anything is
the ten-year run above, which takes a few minutes in the Strategy Backtest page.

### How to read the numbers

An in-sample panel scores a model on the rows it was fitted to; a walk-forward
panel scores it on rows it has never seen. The first is always better. The gap
between them is the most useful thing this project has to show, which is why
the in-sample panel is kept, labelled, underneath the honest one.
""")

st.caption("For educational and research purposes only. Not financial advice.")
