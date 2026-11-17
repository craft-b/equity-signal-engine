"""Pin one Yahoo download so the published numbers can be reproduced.

Yahoo recomputes adjusted prices per request, and two downloads of the same
window differ by fractions of a cent in thousands of cells. The model here is
at a coin flip, so that is enough to change the trade list. The measurement
scripts therefore read docs/snapshot_<ticker>_<period>.csv when it exists.

Re-pinning moves every table in the README: re-run measure_oos.py,
fold_stability.py, sensitivity.py and plot_results.py afterwards, and update
the README from their output.

Usage:
    python scripts/snapshot_data.py            # SPY 10y
    python scripts/snapshot_data.py QQQ 5y
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stock_dashboard"))

from data_pipeline import load_snapshot, write_snapshot  # noqa: E402


def main(ticker: str = "SPY", period: str = "10y") -> int:
    path = write_snapshot(ticker, period)
    data = load_snapshot(path, ticker)
    print(f"wrote {path.relative_to(ROOT)} · {len(data)} bars · "
          f"{data['Date'].min().date()} to {data['Date'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
