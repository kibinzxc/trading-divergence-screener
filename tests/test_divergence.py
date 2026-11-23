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

    def test_signal_goes_stale_after_48_hours_on_4h_and_12_on_1h(self):
        self.assertEqual((m.TF_CFG["4h"]["fresh_h"], m.TF_CFG["1h"]["fresh_h"]), (48, 12))
        self.assertNotIn("fresh", m.TF_CFG["4h"])

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
        # no context yet: scored on the 70 pattern points alone (46.5 / 70)
        self.assertEqual((s["score"], s["grade"], s["regime"], s["partial"], s["tags"]),
                         (66, "B", "COUNTER TREND", True, ["W-PATTERN", "PARTIAL DATA"]))
        self.assertEqual((s["targets_from"], s["targets"]["structure"]), ("signal", 101.0))
        for k in ("targets", "risk_pct", "current_price", "signal_time", "pivot1_time"):
            self.assertIn(k, s)
        self.assertNotIn("_b", s)

    def test_4h_signal_is_stale_after_48_hours(self):
        self.assertEqual([s["fresh"] for s in self.run_tf(wave_df(LONG[:-1] + [(14, 1, 0)]), "4h")], [False])   # 13 bars, 52h
        self.assertEqual([s["fresh"] for s in self.run_tf(wave_df(LONG[:-1] + [(13, 1, 0)]), "4h")], [True])    # 12 bars, 48h

    def test_1h_signal_is_stale_after_12_hours(self):
        out = self.run_tf(wave_df(LONG[:-1] + [(14, 1, 0)]))     # 108 bars, 13 bars after the signal
        self.assertEqual([(s["bars_ago"], s["fresh"]) for s in out], [(13, False)])
        out = self.run_tf(wave_df(LONG[:-1] + [(13, 1, 0)]))
        self.assertEqual([(s["bars_ago"], s["fresh"]) for s in out], [(12, True)])

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
    CASES = [(FULL, True, -3.0, 0.0, "BULL"), (BARE,), (BARE, True, -3.0, 0.0), (FULL, None, None, None, "BULL"),
             (FULL, True, 9.0, 0.02, "BULL"), (BARE, False, 1.0, 0.0, "CHOP"),
             (dict(BARE, price_atr=0.4, macd_frac=0.2, bump_frac=0.4, wick=True), True, 9.0, 0.02, "BEAR"),
             (dict(FULL, side="bearish"), None, -3.0, 0.01, None)]

    def test_parts_add_up_to_the_score_for_full_and_partial_data(self):
        for args in self.CASES:
            parts, (score, _, partial) = m.score_parts(*args), m.score_signal(*args)
            got = sum(p["pts"] for p in parts if p["checked"])
            avail = sum(p["max"] for p in parts if p["checked"])
            self.assertEqual(int(max(0, min(100, round(got / avail * 100)))), score, args)
            self.assertEqual(avail < 100, partial, args)
            self.assertEqual(sum(p["max"] for p in parts), 100)

    def test_parts_are_named_grouped_and_say_what_was_not_checked(self):
        parts = m.score_parts(BARE, True, None, 0.0, "BULL")
        self.assertEqual([(p["key"], p["group"], p["max"]) for p in parts],
                         [("price_move", "pattern", 12), ("macd", "pattern", 12), ("bump", "pattern", 8),
                          ("zero", "pattern", 10), ("vol_dry", "pattern", 10), ("wick", "pattern", 10),
                          ("triple", "pattern", 8), ("htf", "context", 10), ("oi", "context", 10),
                          ("btc", "context", 10)])
        by = {p["key"]: p for p in parts}
        self.assertEqual((by["oi"]["checked"], by["oi"]["pts"]), (False, 0))        # no OI change: not checked
        self.assertEqual((by["htf"]["checked"], by["htf"]["pts"], by["btc"]["pts"]), (True, 10, 10))
        self.assertTrue(all(p["checked"] for p in parts if p["group"] == "pattern"))
        self.assertFalse(any(p["checked"] for p in m.score_parts(BARE) if p["group"] == "context"))

    def test_the_oi_penalty_is_a_part_worth_minus_ten(self):
        by = {p["key"]: p for p in m.score_parts(FULL, True, 9.0, 0.02, "BULL")}
        self.assertEqual((by["oi"]["pts"], by["oi"]["max"], by["oi"]["checked"]), (-10, 10, True))
        self.assertEqual({p["key"]: p for p in m.score_parts(FULL, True, 3.0, 0.02, "BULL")}["oi"]["pts"], 0)
        self.assertEqual({p["key"]: p for p in m.score_parts(FULL, True, -3.0, 0.0, "BULL")}["oi"]["pts"], 10)

    def test_everything_true_is_100(self):
        self.assertEqual(m.score_signal(FULL, True, -3.0, 0.0, "BULL"), (100, "A", False))

    def test_pattern_size_points_top_out_at_32_of_70(self):
        self.assertEqual(m.score_signal(BARE, False, 1.0, 0.0, "CHOP"), (32, "C", False))

    def test_unavailable_context_is_removed_and_the_rest_rescaled(self):
        self.assertEqual(m.score_signal(BARE), (46, "C", True))                         # 32 / 70
        self.assertEqual(m.score_signal(BARE, True, -3.0, 0.0), (58, "B", True))        # 52 / 90
        self.assertEqual(m.score_signal(BARE, True, -3.0, 0.0, "BEAR"), (52, "C", False))  # 52 / 100
        self.assertEqual(m.score_signal(FULL, None, None, None, "BULL"), (100, "A", True))

    def test_btc_points_need_the_btc_state_on_the_signal_side(self):
        self.assertEqual(m.score_signal(FULL, True, -3.0, 0.0, "CHOP")[0], 90)
        self.assertEqual(m.score_signal(FULL, True, -3.0, 0.0, "BEAR")[0], 90)
        self.assertEqual(m.score_signal(dict(FULL, side="bearish"), True, -3.0, 0.0, "BEAR")[0], 100)

    def test_oi_rising_against_the_signal_costs_10(self):
        self.assertEqual(m.score_signal(FULL, True, 9.0, 0.02, "BULL")[0], 80)      # 90 without OI
        self.assertEqual(m.score_signal(dict(FULL, side="bearish"), True, 9.0, -0.02, "BEAR")[0], 80)
        self.assertEqual(m.score_signal(FULL, True, 3.0, 0.02, "BULL")[0], 90)      # under 5%: no penalty
        flat = dict(BARE, price_atr=0.0, macd_frac=0.0, bump_frac=0.25)
        self.assertEqual(m.score_signal(flat, False, 9.0, 0.02, "CHOP")[0], 0)      # clipped at 0

    def test_bearish_oi_bonus_needs_funding_at_or_above_zero(self):
        bear = dict(FULL, side="bearish")
        self.assertEqual(m.score_signal(bear, True, -3.0, 0.01, "BEAR")[0], 100)
        self.assertEqual(m.score_signal(bear, True, -3.0, -0.01, "BEAR")[0], 90)

    def test_grade_cutoffs(self):
        self.assertEqual(m.score_signal(dict(BARE, zero_ok=True, wick=True), True, 1.0, 0.0, "BULL")[1], "A")  # 72
        self.assertEqual(m.score_signal(dict(BARE, zero_ok=True), True, -3.0, 0.0, "CHOP")[1], "B")          # 62
        self.assertEqual(m.score_signal(BARE, True, -3.0, 0.0, "CHOP")[1], "C")                              # 52


class Regime(unittest.TestCase):
    def test_btc_state_needs_rsi_and_the_200_ema_to_agree(self):
        self.assertEqual(m.btc_state(60.0, 110.0, 100.0), "BULL")
        self.assertEqual(m.btc_state(40.0, 90.0, 100.0), "BEAR")
        for args in ((60.0, 90.0, 100.0), (40.0, 110.0, 100.0), (55.0, 110.0, 100.0), (50.0, 90.0, 100.0)):
            self.assertEqual(m.btc_state(*args), "CHOP", args)

    def test_btc_state_is_none_when_any_input_is_missing(self):
        self.assertIsNone(m.btc_state(None, 110.0, 100.0))
        self.assertIsNone(m.btc_state(60.0, 110.0, float("nan")))

    def test_with_trend_needs_btc_and_the_coin_to_agree(self):
        self.assertEqual(m.regime_tag("bullish", True, "BULL"), "WITH TREND")
        self.assertEqual(m.regime_tag("bullish", False, "BULL"), "COUNTER TREND")
        self.assertEqual(m.regime_tag("bullish", True, "BEAR"), "COUNTER TREND")
        self.assertEqual(m.regime_tag("bearish", False, "BEAR"), "WITH TREND")
        self.assertEqual(m.regime_tag("bearish", True, "BEAR"), "COUNTER TREND")

    def test_btc_chop_is_chop(self):
        self.assertEqual(m.regime_tag("bullish", True, "CHOP"), "CHOP")
        self.assertEqual(m.regime_tag("bearish", True, "CHOP"), "CHOP")

    def test_unknown_btc_is_counter_trend(self):
        self.assertEqual(m.regime_tag("bullish", True, None), "COUNTER TREND")


if __name__ == "__main__":
    unittest.main()
