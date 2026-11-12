"""Label construction: censored rows must be dropped, never labelled.

Five private copies of add_target labelled the last `horizon` rows of every
frame 0, because a NaN forward return compared against the threshold is False
rather than NaN and the dropna that was meant to remove them therefore removed
nothing. One shared definition replaces them; these pin its contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labels import add_target


def _rising(n: int) -> pd.DataFrame:
    """Every day up, so every observable label is 1."""
    return pd.DataFrame({"Close": np.linspace(100.0, 100.0 + n, n)})


class TestCensoredRowsAreDropped:
    @pytest.mark.parametrize("horizon", [1, 5, 10])
    def test_exactly_horizon_rows_are_removed(self, horizon):
        n = 40
        out = add_target(_rising(n), horizon, 0.5)
        assert len(out) == n - horizon

    def test_no_row_carries_a_nan_future_return(self):
        out = add_target(_rising(40), 5, 0.5)
        assert out["Future_Return"].notna().all()

    def test_a_market_that_only_rose_has_no_negative_labels(self):
        """The regression. The old copies produced `horizon` zeros here."""
        out = add_target(_rising(40), 5, 0.01)
        assert (out["Target"] == 1).all(), (
            f"found {int((out['Target'] == 0).sum())} rows labelled 0 in a "
            "series that rose every day — censored rows were labelled instead "
            "of dropped"
        )


class TestLabelSemantics:
    def test_threshold_is_in_percent(self):
        # +1% per bar over 1 bar: a 0.5% threshold passes, a 2% threshold fails.
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.01]})
        assert add_target(df, 1, 0.5)["Target"].tolist() == [1, 1]
        assert add_target(df, 1, 2.0)["Target"].tolist() == [0, 0]

    def test_falling_market_labels_zero(self):
        df = pd.DataFrame({"Close": np.linspace(100.0, 80.0, 30)})
        out = add_target(df, 5, 0.5)
        assert (out["Target"] == 0).all()

    def test_index_is_reset(self):
        out = add_target(_rising(20), 3, 0.5)
        assert list(out.index) == list(range(len(out)))

    def test_input_is_not_mutated(self):
        df = _rising(20)
        before = df.copy()
        add_target(df, 3, 0.5)
        pd.testing.assert_frame_equal(df, before)

    def test_horizon_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="horizon"):
            add_target(_rising(20), 0, 0.5)
