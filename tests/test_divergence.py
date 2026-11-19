"""Tests for pivot detection and divergence geometry.

Run with:  python -m unittest tests.test_divergence -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402


def tri(n, c, depth, w=6):
    """A triangle of height `depth` centred on bar c, zero elsewhere."""
    t = np.arange(n)
    return depth * np.clip(1 - np.abs(t - c) / w, 0, None)


def make_df(a, b, side="bullish", n=120, freq="1h"):
    """Hand-built candles AND indicators with one clean regular divergence.

    Bullish: price low at b is lower than at a, MACD low at b is higher.
    Bearish is the mirror. Indicators are supplied directly so the test
    controls the geometry instead of fighting an EMA.
    """
    sgn = -1 if side == "bullish" else 1
    ext = 100 + sgn * (tri(n, a, 5) + tri(n, b, 8))
    macd = sgn * (tri(n, a, 2.0) + tri(n, b, 1.0))
    if side == "bullish":
        low, high = ext, ext + 1
    else:
        high, low = ext, ext - 1
    idx = pd.date_range("2026-01-01", periods=n, freq=freq)
    return pd.DataFrame({"open": low + .5, "high": high, "low": low, "close": low + .5,
                         "volume": 1.0, "macd": macd, "signal": macd - sgn * 0.1,
                         "atr": 1.0}, index=idx)


class Pivots(unittest.TestCase):
    VALS = np.array([10.0] * 10 + [5.0] + [10.0] * 3)   # low at bar 10, 3 bars after it

    def test_pivot_is_confirmed_after_rb_bars(self):
        self.assertEqual(m.find_pivots(self.VALS, 5, 3)[1], [10])

    def test_default_right_window_equals_left(self):
        self.assertEqual(m.find_pivots(self.VALS, 5)[1], [])

    def test_pivot_does_not_move_when_bars_are_appended(self):
        longer = np.append(self.VALS, [10.0] * 5)
        self.assertEqual(m.find_pivots(longer, 5, 3)[1], [10])


class Config(unittest.TestCase):
    def test_timeframes_are_4h_then_1h(self):
        self.assertEqual(list(m.TF_CFG), ["4h", "1h"])

    def test_loose_is_gone(self):
        self.assertEqual(sorted(m.STRICTNESS), ["balanced", "strict"])


class Detection(unittest.TestCase):
    RULES = m.STRICTNESS["balanced"]

    def detect(self, a, b, side, tf):
        return m.detect_divergences(make_df(a, b, side), m.TF_CFG[tf], self.RULES)

    def test_clear_divergence_is_found_on_both_timeframes(self):
        for tf in ("4h", "1h"):
            for side in ("bullish", "bearish"):
                out = self.detect(60, 100, side, tf)
                self.assertEqual([(s["side"], s["pivot_dist"]) for s in out],
                                 [(side, 40)], (tf, side))

    def test_listed_three_bars_after_the_second_pivot(self):
        out = self.detect(70, 116, "bullish", "4h")     # 120 bars: 3 bars follow bar 116
        self.assertEqual([(s["bars_ago"], s["fresh"]) for s in out], [(3, True)])

    def test_second_pivot_older_than_max_age_is_dropped(self):
        for tf in ("4h", "1h"):
            self.assertEqual(self.detect(20, 60, "bullish", tf), [])   # 59 bars old

    def test_pivots_further_apart_than_max_dist_are_dropped(self):
        for tf in ("4h", "1h"):
            self.assertEqual(self.detect(20, 100, "bearish", tf), [])  # 80 bars apart


if __name__ == "__main__":
    unittest.main()
