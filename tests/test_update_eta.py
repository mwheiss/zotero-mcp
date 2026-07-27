"""Tests for median-based update-db ETA estimation and display formatting."""

import io

import pytest

import zotero_mcp.semantic_search as semantic_search
from zotero_mcp.semantic_search import (
    _CumulativeETA,
    _display_width,
    _format_eta,
    _MedianETA,
    _truncate_display,
    _write_progress_line,
)


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


def test_display_width_counts_cjk_and_combining_characters():
    assert _display_width("abc") == 3
    assert _display_width("二宮") == 4
    assert _display_width("e\u0301") == 1


def test_display_truncation_respects_wide_character_columns():
    result = _truncate_display("ab二宮cd", 7)

    assert result == "ab二..."
    assert _display_width(result) == 7


class _TTY(io.StringIO):
    def isatty(self):
        return True

    def fileno(self):
        raise OSError


def test_tty_progress_clears_whole_line_before_each_repaint(monkeypatch):
    stream = _TTY()
    monkeypatch.setattr(semantic_search, "_terminal_columns", lambda _stream: 20)

    _write_progress_line(stream, "long Japanese title 日本語")
    _write_progress_line(stream, "short")

    repaints = stream.getvalue().split("\r\x1b[2K")
    assert len(repaints) == 3
    assert _display_width(repaints[1]) <= 19
    assert repaints[2] == "short"


def test_non_tty_progress_pads_over_stale_tail(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(semantic_search, "_terminal_columns", lambda _stream: 12)

    _write_progress_line(stream, "long title")
    _write_progress_line(stream, "short")

    assert stream.getvalue().endswith("\rshort" + " " * 6)
