"""Tests for the outside context a signal is graded with.

Run with:  python -m unittest tests.test_context -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import LONG, wave_df  # noqa: E402

T = pd.Timestamp


class HigherTimeframe(unittest.TestCase):
    D = pd.DataFrame({"macd": [1.0, 2.0, 1.5]}, index=pd.date_range("2026-01-01", periods=3, freq="D"))

    def test_uses_only_bars_completed_by_the_signal_close(self):
        # at Jan 3 00:00 the Jan 1 and Jan 2 bars are complete: 1.0 -> 2.0 rising
        self.assertTrue(m.htf_agrees(self.D, T("2026-01-03"), 1440, "bullish"))
        # a day later the Jan 3 bar counts too: 2.0 -> 1.5 falling
        self.assertFalse(m.htf_agrees(self.D, T("2026-01-04"), 1440, "bullish"))
        self.assertTrue(m.htf_agrees(self.D, T("2026-01-04"), 1440, "bearish"))

    def test_too_little_history_or_none_is_no_agreement(self):
        self.assertFalse(m.htf_agrees(self.D, T("2026-01-02"), 1440, "bullish"))
        self.assertFalse(m.htf_agrees(None, T("2026-01-04"), 1440, "bullish"))

    def test_asof(self):
        s = self.D["macd"]
        self.assertEqual((m.asof(s, T("2026-01-02 12:00")), m.asof(s, T("2025-12-31"))), (2.0, None))


def ltf_frame():
    """40 bars of 15m: swing high 105 at bar 10, MACD crosses up at bar 20,
    first close above the swing high at bar 24."""
    n = 40
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    high = np.full(n, 101.0); high[10] = 105; high[24:] = 107
    close = np.full(n, 100.0); close[24:] = 106
    d = np.full(n, -1.0); d[20:] = 1
    return pd.DataFrame({"high": high, "low": 99.0, "close": close, "macd": d, "signal": 0.0},
                        index=idx)


class Trigger(unittest.TestCase):
    def test_fires_on_the_close_above_the_swing_high_after_the_cross(self):
        f = ltf_frame()
        close_ms = int(f.index[25].timestamp() * 1000)          # bar 24 closes when bar 25 opens
        self.assertEqual(m.ltf_trigger(f, "bullish", f.index[5], f.index[18], 15), (106.0, close_ms))

    def test_a_cross_before_the_signal_bar_does_not_count(self):
        f = ltf_frame()
        self.assertIsNone(m.ltf_trigger(f, "bullish", f.index[5], f.index[22], 15))

    def test_the_swing_high_must_be_after_price_low_b(self):
        f = ltf_frame()
        self.assertIsNone(m.ltf_trigger(f, "bullish", f.index[12], f.index[18], 15))

    def test_bearish_is_the_mirror(self):
        f = ltf_frame()
        mir = pd.DataFrame({"high": 201.0, "low": 200 - f["high"], "close": 200 - f["close"],
                            "macd": -f["macd"], "signal": 0.0}, index=f.index)
        self.assertEqual(m.ltf_trigger(mir, "bearish", f.index[5], f.index[18], 15)[0], 94.0)


class OpenInterest(unittest.TestCase):
    HIST = [{"timestamp": 1000, "openInterestAmount": 100},
            {"timestamp": 2000, "openInterestAmount": 90}]

    def test_change_between_two_times(self):
        self.assertAlmostEqual(m.oi_change_pct(self.HIST, 1500, 2500), -10.0)

    def test_none_when_history_does_not_cover_both_times(self):
        self.assertIsNone(m.oi_change_pct(self.HIST, 500, 2500))      # nothing at or before t0
        self.assertIsNone(m.oi_change_pct(self.HIST, 1500, 1800))     # same point for both
        self.assertIsNone(m.oi_change_pct(None, 1500, 2500))

    def test_falls_back_to_usd_value(self):
        hist = [{"timestamp": 1000, "openInterestValue": 50.0},
                {"timestamp": 2000, "openInterestValue": 60.0}]
        self.assertAlmostEqual(m.oi_change_pct(hist, 1000, 2000), 20.0)


class AddContext(unittest.TestCase):
    def setUp(self):
        self.sig = m.signals_for_tf(wave_df(LONG), "1h", m.STRICTNESS["balanced"],
                                    100.5, "ENS/USDT:USDT", 9e7)[0]
        idx = pd.date_range("2025-12-25", periods=80, freq="4h")
        self.h4 = pd.DataFrame({"macd": np.arange(80.0)}, index=idx)        # rising
        self.btc = pd.DataFrame({"rsi": 60.0}, index=idx)

    def test_without_context_it_is_the_bare_structure_grade(self):
        self.assertEqual((self.sig["score"], self.sig["grade"], self.sig["regime"]),
                         (53, "C", "COUNTER TREND"))

    def test_context_lifts_the_grade_and_sets_the_regime(self):
        s = self.sig
        hist = [{"timestamp": s["b_start_ms"] - 1, "openInterestAmount": 100},
                {"timestamp": s["b_ms"] - 1, "openInterestAmount": 90}]
        m.add_context(s, self.h4, None, hist, 0.0, self.btc)
        self.assertEqual((s["score"], s["grade"], s["regime"], s["htf_ok"], s["oi_chg_pct"]),
                         (78, "A", "WITH TREND", True, -10.0))
        self.assertEqual((s["tags"], s["trigger_price"], s["targets_from"]),
                         (["W-PATTERN"], None, "signal"))

    def test_missing_context_changes_nothing(self):
        m.add_context(self.sig, None, None, None, None, None)
        self.assertEqual((self.sig["score"], self.sig["regime"]), (53, "COUNTER TREND"))


def trend_frame(freq, direction):
    idx = pd.date_range("2025-12-01", periods=900, freq=freq)
    return pd.DataFrame({"macd": direction * np.arange(900.0)}, index=idx)


def ltf_for(sig):
    """15m frame whose MACD crosses up 2 bars after the signal bar opens and
    whose close clears the swing high (bar 10) 4 bars later. It starts at B."""
    n, i0 = 70, int((sig["signal_ms"] - sig["b_ms"]) // 900_000)
    high = np.full(n, 101.0); high[10] = 105; high[i0 + 6:] = 107
    close = np.full(n, 100.0); close[i0 + 6:] = 106
    d = np.full(n, -1.0); d[i0 + 2:] = 1
    return pd.DataFrame({"high": high, "low": 99.0, "close": close, "macd": d, "signal": 0.0},
                        index=pd.date_range(pd.Timestamp(sig["b_ms"], unit="ms"), periods=n, freq="15min"))


class GradeSignals(unittest.TestCase):
    R = m.STRICTNESS["balanced"]
    UP, DOWN = trend_frame("4h", 1), trend_frame("4h", -1)
    UP_D, DOWN_D = trend_frame("D", 1), trend_frame("D", -1)

    def sig(self, tf="1h", n=97, **kw):
        df = wave_df(LONG, freq="4h" if tf == "4h" else "1h")
        return m.signals_for_tf(df.iloc[:n], tf, self.R, 100.5, "ENS/USDT:USDT", 9e7, **kw)[0]

    def grade(self, s, df4=None, df1=None, d1=None, df15=None, oi_d=None, oi_h=None, btc=None):
        return m.grade_signals([s], df4, df1, d1, df15, oi_d, oi_h, 0.0, btc)[0]

    def test_a_4h_signal_takes_its_higher_timeframe_from_d1_not_df4(self):
        self.assertTrue(self.grade(self.sig("4h"), df4=self.DOWN, d1=self.UP_D)["htf_ok"])
        self.assertFalse(self.grade(self.sig("4h"), df4=self.UP, d1=self.DOWN_D)["htf_ok"])

    def test_a_1h_signal_takes_its_higher_timeframe_from_df4_not_d1(self):
        self.assertTrue(self.grade(self.sig(), df4=self.UP, d1=self.DOWN_D)["htf_ok"])
        self.assertFalse(self.grade(self.sig(), df4=self.DOWN, d1=self.UP_D)["htf_ok"])

    def test_a_1h_fresh_signal_takes_its_trigger_from_the_15m_frame(self):
        s = self.sig()
        self.assertTrue(s["fresh"])
        ok = self.grade(self.sig(), df15=ltf_for(s))
        self.assertIsNotNone(ok["trigger_price"])
        self.assertIn("LTF TRIGGER", ok["tags"])
        wrong = self.grade(self.sig(), df1=ltf_for(s))            # 15m frame in the 1H slot
        self.assertIsNone(wrong["trigger_price"])
        self.assertNotIn("LTF TRIGGER", wrong["tags"])

    def test_oi_uses_hourly_when_it_covers_wave_b_else_daily(self):
        s = self.sig()
        def hist(change):
            return [{"timestamp": s["b_start_ms"] - 1, "openInterestAmount": 100},
                    {"timestamp": s["b_ms"] - 1, "openInterestAmount": 100 + change}]
        hourly, daily = hist(-10), hist(20)
        self.assertEqual(self.grade(self.sig(), oi_d=daily, oi_h=hourly)["oi_chg_pct"], -10.0)
        self.assertEqual(self.grade(self.sig(), oi_d=daily, oi_h=None)["oi_chg_pct"], 20.0)
        self.assertEqual(self.grade(self.sig(), oi_d=daily, oi_h=[{"timestamp": 1, "openInterestAmount": 5}])["oi_chg_pct"], 20.0)

    def test_forming_signal_with_a_trigger_is_trigger_first_and_a_yes_early(self):
        df = wave_df(LONG)
        s = self.sig(n=94, forming_df=df.iloc[:95])
        self.assertTrue(s["forming"])
        btc = pd.DataFrame({"rsi": 60.0}, index=pd.date_range("2025-12-01", periods=900, freq="4h"))
        hist = [{"timestamp": s["b_start_ms"] - 1, "openInterestAmount": 100},
                {"timestamp": s["b_ms"] - 1, "openInterestAmount": 90}]
        s = self.grade(s, df4=self.UP, df15=ltf_for(s), oi_h=hist, btc=btc)
        self.assertTrue(s["trigger_first"])
        self.assertEqual(s["grade"], "A")
        self.assertTrue(m.entry_verdict(s)["reason"].startswith("Yes (early)"))


if __name__ == "__main__":
    unittest.main()
