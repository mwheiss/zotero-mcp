"""Tests for median-based update-db ETA estimation and display formatting."""

import pytest

from zotero_mcp.semantic_search import _CumulativeETA, _format_eta, _MedianETA


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


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


def test_cumulative_eta_accounts_for_slow_extraction_tail():
    clock = _Clock()
    eta = _CumulativeETA(total=100, clock=clock)

    # Nine cache hits are nearly free, but one real extraction is slow.
    clock.now = 10
    assert eta.estimate(10) == pytest.approx(90)

    # Later fast cache hits do not erase the elapsed extraction cost.
    clock.now = 11
    assert eta.estimate(50) == pytest.approx(11)


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
