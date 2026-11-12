"""Render the results figure for the README.

Generated from committed data by committed code, so the picture cannot drift
away from the numbers the way a hand-taken screenshot does. Reads
``docs/sensitivity_*.csv`` produced by ``scripts/sensitivity.py``.

Usage:
    python scripts/plot_results.py                       # SPY 10y
    python scripts/plot_results.py docs/sensitivity_qqq_5y.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "docs" / "sensitivity_spy_10y.csv"

INK = "#1a1d21"
MUTED = "#6b7280"
HAIRLINE = "#d8dbe0"
# One neutral colour on purpose. Nothing in this grid clears what chance
# would produce, so colouring by sign would dress noise up as winners and
# losers.
BAR = "#8b9099"


def main(csv: str | None = None) -> int:
    path = Path(csv) if csv else DEFAULT_CSV
    if not path.exists():
        print(f"missing {path} — run scripts/sensitivity.py first")
        return 1

    frame = pd.read_csv(path).sort_values("sharpe").reset_index(drop=True)
    labels = [
        f"{int(r.holding)}d · {r.threshold:g}% · {r.min_confidence:g}"
        for r in frame.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    ax.barh(labels, frame["sharpe"], color=BAR, height=0.68)

    ax.axvline(0, color=INK, linewidth=1.1, zorder=3)
    ax.set_xlabel("Out-of-sample Sharpe ratio", fontsize=10, color=INK)
    significant = int((frame["p_value"] < 0.05).sum())
    expected = 0.05 * len(frame)
    # The title is read from the data. It used to say "Every configuration
    # lands on zero", which was true of the run that wrote it and false of the
    # next one, where a single cell cleared p<0.05 -- the one false positive a
    # grid this size is expected to produce. A title that cannot change with
    # the numbers under it is a caption waiting to become a lie.
    if significant == 0:
        title = "Every configuration lands on zero"
    elif significant <= round(expected):
        title = f"{significant} of {len(frame)} clears p<0.05 — about what chance gives"
    else:
        title = f"{significant} of {len(frame)} clear p<0.05 — more than chance gives"
    ax.set_title(title, fontsize=15, color=INK, loc="left", pad=34, weight="600")
    ax.text(
        0, 1.055,
        f"SPY · 10 years · walk-forward with embargo · {len(frame)} combinations of "
        f"holding period, signal threshold and confidence cut",
        transform=ax.transAxes, fontsize=9, color=MUTED,
    )
    ax.text(
        0, 1.012,
        f"{significant} of {len(frame)} significant at p<0.05 — "
        f"chance alone would give about {expected:.1f}",
        transform=ax.transAxes, fontsize=9, color=MUTED,
    )

    ax.tick_params(axis="y", labelsize=7.6, colors=MUTED, length=0)
    ax.tick_params(axis="x", labelsize=8.6, colors=MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.grid(axis="x", color=HAIRLINE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.text(
        0.995, -0.135,
        "holding · threshold · min confidence",
        transform=ax.transAxes, ha="right", fontsize=7.6, color=MUTED,
    )

    fig.tight_layout()
    out = ROOT / "docs" / "results_sensitivity.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print(f"  sharpe range {frame['sharpe'].min():.2f} to {frame['sharpe'].max():.2f}")
    print(f"  significant at p<0.05: {significant} of {len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
