"""Tests for pivots, config, the signal record, the score and the regime tag.

Run with:  python -m unittest tests.test_divergence -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import LONG, wave_df  # noqa: E402

RULES = m.STRICTNESS["balanced"]


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

    def test_signal_goes_stale_after_6_bars_on_4h_and_12_on_1h(self):
        self.assertEqual((m.TF_CFG["4h"]["fresh"], m.TF_CFG["1h"]["fresh"]), (6, 12))

    def test_the_pivot_detector_is_gone(self):
        for name in ("detect_divergences", "swing_pivots", "setup_score", "PIVOT_SYNC"):
            self.assertFalse(hasattr(m, name), name)


class SignalsForTf(unittest.TestCase):
    def run_tf(self, df, tf="1h", **kw):
        return m.signals_for_tf(df, tf, RULES, 100.5, "ENS/USDT:USDT", 9e7, **kw)

    def test_finished_signal_carries_everything_a_row_needs(self):
        out = self.run_tf(wave_df(LONG))            # 102 bars, signal bar 94
        self.assertEqual(len(out), 1)
        s = out[0]
        self.assertEqual((s["tf"], s["pair"], s["base"], s["side"], s["weekly_volume"]),
                         ("1h", "ENS/USDT:USDT", "ENS", "bullish", 9e7))
        self.assertEqual((s["bars_ago"], s["fresh"], s["forming"], s["status"]),
                         (7, True, False, "active"))
        self.assertEqual((s["signal_price"], s["stop_price"], s["pivot2_price"]),
                         (100.5, 91.8184, 92.0684))     # stop = low B - 0.25 ATR
        self.assertEqual((s["score"], s["grade"], s["regime"], s["tags"]),
                         (53, "C", "COUNTER TREND", ["W-PATTERN"]))
        self.assertEqual((s["targets_from"], s["targets"]["structure"]), ("signal", 101.0))
        for k in ("targets", "risk_pct", "current_price", "signal_time", "pivot1_time"):
            self.assertIn(k, s)
        self.assertNotIn("_b", s)

    def test_same_signal_is_stale_on_4h(self):
        self.assertEqual([s["fresh"] for s in self.run_tf(wave_df(LONG), "4h")], [False])

    def test_too_few_bars_gives_nothing(self):
        self.assertEqual(self.run_tf(wave_df(LONG).iloc[:60]), [])

    def test_signal_bar_still_forming_is_provisional(self):
        df = wave_df(LONG)
        closed, with_open = df.iloc[:94], df.iloc[:95]     # bar 94 is the closing cross
        self.assertEqual(self.run_tf(closed), [])
        out = self.run_tf(closed, forming_df=with_open)
        self.assertEqual([(s["forming"], s["bars_ago"]) for s in out], [(True, 0)])


FULL = dict(side="bullish", price_atr=2.0, macd_frac=0.5, bump_frac=0.75,
            zero_ok=True, vol_dry=True, wick=True, triple=True)
BARE = dict(FULL, zero_ok=False, vol_dry=False, wick=False, triple=False)


class Score(unittest.TestCase):
    def test_everything_true_is_100(self):
        self.assertEqual(m.score_signal(FULL, True, -3.0, 0.0), (100, "A"))

    def test_structure_alone_tops_out_at_40(self):
        self.assertEqual(m.score_signal(BARE), (40, "C"))

    def test_oi_rising_against_the_signal_costs_10(self):
        self.assertEqual(m.score_signal(FULL, True, 9.0, 0.02)[0], 80)      # 90 without OI
        self.assertEqual(m.score_signal(dict(FULL, side="bearish"), True, 9.0, -0.02)[0], 80)
        self.assertEqual(m.score_signal(FULL, True, 3.0, 0.02)[0], 90)      # under 5%: no penalty

    def test_bearish_oi_bonus_needs_funding_at_or_above_zero(self):
        bear = dict(FULL, side="bearish")
        self.assertEqual(m.score_signal(bear, True, -3.0, 0.01)[0], 100)
        self.assertEqual(m.score_signal(bear, True, -3.0, -0.01)[0], 90)

    def test_grade_cutoffs(self):
        self.assertEqual(m.score_signal(dict(BARE, zero_ok=True, wick=True), True)[1], "A")   # 75
        self.assertEqual(m.score_signal(BARE, True)[1], "B")                                  # 55
        self.assertEqual(m.score_signal(dict(BARE, zero_ok=True))[1], "C")                    # 50


class Regime(unittest.TestCase):
    def test_with_trend_needs_btc_and_the_coin_to_agree(self):
        self.assertEqual(m.regime_tag("bullish", True, 60.0), "WITH TREND")
        self.assertEqual(m.regime_tag("bullish", False, 60.0), "COUNTER TREND")
        self.assertEqual(m.regime_tag("bullish", True, 40.0), "COUNTER TREND")
        self.assertEqual(m.regime_tag("bearish", False, 40.0), "WITH TREND")
        self.assertEqual(m.regime_tag("bearish", True, 40.0), "COUNTER TREND")

    def test_unknown_btc_is_counter_trend(self):
        self.assertEqual(m.regime_tag("bullish", True, None), "COUNTER TREND")


if __name__ == "__main__":
    unittest.main()
