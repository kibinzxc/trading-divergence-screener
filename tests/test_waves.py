"""Tests for MACD crossover waves and wave divergence.

Run with:  python -m unittest tests.test_waves -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import W, wave_df  # noqa: E402

RULES = {"min_price_pct": 0.25, "atr_frac": 0.20, "min_macd_frac": 0.15, "zero_line": False}
CROSSES = [(10, -1), (22, 1), (32, -1), (44, 1)]


def crosses(df):
    return m.macd_crosses(df["macd"].values, df["signal"].values)


def detect(lobes, side="bullish", rules=RULES, **kw):
    return m.detect_wave_divergence(wave_df(lobes, side, **kw), side, rules)


class Indicators(unittest.TestCase):
    def test_ema200_column(self):
        df = wave_df(W)[["open", "high", "low", "close", "volume"]]
        self.assertIn("ema200", m.compute_indicators(df).columns)

    def test_rsi_extremes(self):
        up = pd.Series(np.arange(1.0, 60))
        self.assertEqual((m.rsi(up).iloc[-1], m.rsi(100 - up).iloc[-1]), (100.0, 0.0))


class Crosses(unittest.TestCase):
    def test_one_cross_per_lobe_boundary(self):
        self.assertEqual(crosses(wave_df(W)), CROSSES)

    def test_tiny_double_dot_does_not_split_a_trough(self):
        df = wave_df(W)
        df.iloc[15, df.columns.get_loc("macd")] = 0.01     # pokes above signal for one bar
        self.assertEqual(crosses(df), CROSSES)

    def test_wave_shorter_than_three_bars_is_merged(self):
        df = wave_df(W)
        df.iloc[15:17, df.columns.get_loc("macd")] = 1.0   # a decisive but 2-bar bull wave
        self.assertEqual(crosses(df), CROSSES)


class Detection(unittest.TestCase):
    def test_w_shape_is_found_on_its_own_side_only(self):
        for side, other in (("bullish", "bearish"), ("bearish", "bullish")):
            df = wave_df(W, side)
            out = m.detect_wave_divergence(df, side, RULES)
            self.assertEqual([(d["a_m"], d["b_m"], d["bump_bar"], d["signal_bar"]) for d in out],
                             [(15, 37, 26, 44)], side)
            self.assertEqual(out[0]["crosses"], crosses(df))
            self.assertEqual(m.detect_wave_divergence(df, other, RULES), [], side)

    def test_levels_come_from_wicks_and_the_macd_line(self):
        d = detect(W)[0]
        self.assertEqual((round(d["a_price"], 2), round(d["b_price"], 2)), (95.04, 92.07))
        self.assertEqual((round(d["a_macd"], 2), round(d["b_macd"], 2)), (-1.98, -0.99))
        d = detect(W, "bearish")[0]
        self.assertEqual((round(d["a_price"], 2), round(d["b_price"], 2)), (104.96, 107.93))
        self.assertEqual((round(d["a_macd"], 2), round(d["b_macd"], 2)), (1.98, 0.99))

    def test_not_reported_until_wave_b_is_closed_by_a_cross(self):
        df = wave_df(W)
        self.assertEqual(m.detect_wave_divergence(df.iloc[:44], "bullish", RULES), [])
        self.assertEqual(len(m.detect_wave_divergence(df.iloc[:45], "bullish", RULES)), 1)

    def test_macd_trough_must_be_higher(self):
        self.assertEqual(detect([(10, 1, 0), (12, 1.0, 5), (10, 1.2, 0), (12, 2.0, 8), (8, 1, 0)]), [])

    def test_price_low_must_be_lower(self):
        self.assertEqual(detect([(10, 1, 0), (12, 2.0, 8), (10, 1.2, 0), (12, 1.0, 5), (8, 1, 0)]), [])

    def test_a_bump_too_small_to_count_gives_nothing(self):
        self.assertEqual(detect([(10, 1, 0), (12, 6.0, 5), (10, 0.2, 0), (12, 1.0, 8), (8, 1, 0)]), [])

    def test_troughs_must_be_8_to_80_bars_apart(self):
        self.assertEqual(detect([(10, 1, 0), (12, 2.0, 5), (80, 1.2, 0), (12, 1.0, 8), (8, 1, 0)]), [])
        self.assertEqual(detect([(10, 1, 0), (4, 2.0, 5), (3, 1.2, 0), (4, 1.0, 8), (8, 1, 0)]), [])

    def test_strict_needs_both_troughs_below_zero(self):
        strict = dict(RULES, zero_line=True)
        self.assertEqual(len(detect(W, rules=strict)), 1)
        self.assertEqual(detect(W, rules=strict, offset=5.0), [])
        self.assertEqual([d["zero_ok"] for d in detect(W, offset=5.0)], [False])

    def test_triple_when_the_trough_before_a_also_diverges(self):
        lobes = [(10, 1, 0), (12, 3.0, 3), (10, 1.2, 0), (12, 2.0, 5), (10, 1.2, 0), (12, 1.0, 8), (8, 1, 0)]
        self.assertEqual([(d["a_m"], d["triple"]) for d in detect(lobes)], [(15, False), (37, True)])

    def test_structure_facts_for_the_score(self):
        d = detect(W)[0]
        self.assertEqual((round(d["bump_frac"], 2), round(d["macd_frac"], 2), round(d["price_atr"], 2)),
                         (0.69, 0.31, 2.97))
        self.assertEqual((d["wick"], d["vol_dry"]), (True, False))
        df = wave_df(W)
        df.iloc[10:22, df.columns.get_loc("volume")] = 3.0          # wave A traded 3x wave B
        self.assertTrue(m.detect_wave_divergence(df, "bullish", RULES)[0]["vol_dry"])


if __name__ == "__main__":
    unittest.main()
