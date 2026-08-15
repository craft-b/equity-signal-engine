"""The Streamlit pages must at least parse.

Nothing else in the suite touches ``stock_dashboard/pages/``: the pages are
only ever executed by Streamlit itself, so a syntax error in one ships green
and only surfaces when someone opens the app. That happened — an edit left an
unterminated string literal in the backtest page and 173 tests still passed.

Compiling each page is cheap and closes that hole. It does not execute them,
so no Streamlit runtime is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "stock_dashboard"
PAGES = sorted(DASHBOARD.glob("pages/*.py")) + [DASHBOARD / "app.py"]


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_page_compiles(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    compile(source, str(path), "exec")


def test_every_page_is_covered() -> None:
    """Guard against the glob silently matching nothing."""
    assert len(PAGES) >= 3, f"expected app.py plus pages, found {PAGES}"
