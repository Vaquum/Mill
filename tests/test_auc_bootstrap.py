import numpy as np
import pytest

from mill import discover
from mill.auc_bootstrap import draws


@pytest.mark.parametrize("n,block", [(1, 24), (59, 24), (317, 23), (101, 500), (400, 1), (20003, 24)])
def test_difference_counts_match_materialized_moving_blocks_exactly(n, block):
    rng = np.random.default_rng(981)
    x = np.round(rng.normal(size=n), 1)
    if n > 10:
        x[::17] = np.nan
    y = rng.integers(0, 2, size=n).astype(float)
    count = (n+block-1)//block
    starts = np.random.default_rng(44).integers(0, n, size=(7, count))
    rows = ((starts[:, :, None]+np.arange(block)) % n).reshape(7, -1)[:, :n]
    weights = np.stack([np.bincount(row, minlength=n) for row in rows]).astype(float)
    expected = discover.weighted_auc(x, y, weights)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    groups = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    codes = np.empty(n, dtype=np.int32)
    codes[order] = np.repeat(np.arange(groups.size), np.diff(np.r_[groups, n]))
    positive_groups = np.unique(codes[y == 1])
    slots = np.full(n, -1, dtype=np.int32)
    slots[y == 1] = np.searchsorted(positive_groups, codes[y == 1])
    actual = draws(codes, slots, positive_groups, starts, block, groups.size)
    np.testing.assert_array_equal(actual, expected)


def test_fast_and_original_intervals_are_identical(monkeypatch):
    rng = np.random.default_rng(414)
    x, y = rng.normal(size=509), (rng.random(509) < .2).astype(float)
    expected = discover.bootstrap_interval(x, y, 24, 81)
    monkeypatch.setattr(discover, "_FAST_AUC_BOOTSTRAP", False)
    assert discover.bootstrap_interval(x, y, 24, 81) == expected
