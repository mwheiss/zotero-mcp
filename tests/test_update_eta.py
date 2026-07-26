"""Tests for median-based update-db ETA estimation and display formatting."""

import pytest

from zotero_mcp.semantic_search import _format_eta, _MedianETA


def test_median_eta_uses_all_completed_entry_durations():
    eta = _MedianETA(total=10)

    assert eta.estimate(0) is None

    eta.record(2)
    assert eta.estimate(1) == pytest.approx(18)

    eta.record(10)
    assert eta.estimate(2) == pytest.approx(48)

    # All observations remain in the sample; the median is now 4 seconds.
    eta.record(4)
    assert eta.estimate(3) == pytest.approx(28)
    assert eta.estimate(10) == 0


def test_median_eta_accounts_for_encoder_parallelism():
    eta = _MedianETA(total=10, parallelism=2)
    eta.record(4)
    eta.record(6)

    assert eta.estimate(2) == pytest.approx(20)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "calculating"),
        (0, "0s"),
        (8.1, "9s"),
        (60, "1m 00s"),
        (3661, "1h 01m"),
        (90000, "1d 1h"),
    ],
)
def test_format_eta(seconds, expected):
    assert _format_eta(seconds) == expected
