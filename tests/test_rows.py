"""Tests for the entry verdict, context and row grouping.

Run with:  python -m unittest tests.test_rows -v
"""
import datetime as dt
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402

SIG = {"status": "active", "fresh": True, "bars_ago": 2, "r_reached": 0.0,
       "pct_chg": 0.4, "signal_price": 100.0, "macd_cross": True,
       "stop_price": 97.0, "risk_pct": 3.0}


def sig(**kw):
    return dict(SIG, **kw)


class EntryVerdict(unittest.TestCase):
    def verdict(self, s, *a):
        return m.entry_verdict(s, *a)["verdict"]

    def test_enter_when_every_check_passes(self):
        v = m.entry_verdict(sig())
        self.assertEqual((v["verdict"], v["level"]), ("enter", 97.0))
        self.assertIn("Risk 3.0%", v["reason"])

    def test_invalidated_is_skip(self):
        v = m.entry_verdict(sig(status="invalidated"))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("hit the stop", v["reason"])

    def test_ran_is_skip(self):
        v = m.entry_verdict(sig(status="ran", r_reached=2.5))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("2.5x the risk", v["reason"])

    def test_not_fresh_is_skip(self):
        v = m.entry_verdict(sig(fresh=False, bars_ago=20))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("20 bars ago", v["reason"])

    def test_extended_waits_for_a_pullback_to_the_signal_price(self):
        v = m.entry_verdict(sig(status="extended", pct_chg=3.1))
        self.assertEqual((v["verdict"], v["level"]), ("wait", 100.0))
        self.assertIn("3.1%", v["reason"])

    def test_no_macd_cross_is_wait(self):
        self.assertEqual(self.verdict(sig(macd_cross=False)), "wait")

    def test_event_inside_24h_is_wait_and_names_the_event(self):
        v = m.entry_verdict(sig(), 14.2, "CPI m/m")
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("CPI m/m is due in 14h", v["reason"])

    def test_event_beyond_24h_or_unknown_does_not_block(self):
        self.assertEqual(self.verdict(sig(), 30.0, "CPI m/m"), "enter")
        self.assertEqual(self.verdict(sig(), None), "enter")

    def test_skip_outranks_wait(self):
        self.assertEqual(self.verdict(sig(status="invalidated"), 2.0, "FOMC"), "skip")


class NextMover(unittest.TestCase):
    NOW = dt.datetime(2026, 11, 19, 12, 0, tzinfo=dt.timezone.utc)
    EVENTS = [
        {"ts": "2026-11-19T10:00:00Z", "mover": True,  "title": "Past print"},
        {"ts": "2026-11-19T15:00:00Z", "mover": False, "title": "Not a mover"},
        {"ts": "2026-11-20T13:30:00Z", "mover": True,  "title": "CPI m/m"},
        {"ts": "2026-11-21T19:00:00Z", "mover": True,  "title": "FOMC Statement"},
    ]

    def test_nearest_future_mover(self):
        self.assertEqual(m.next_mover(self.EVENTS, self.NOW), (25.5, "CPI m/m"))

    def test_none_when_nothing_is_scheduled(self):
        self.assertEqual(m.next_mover([], self.NOW), (None, ""))


def oi_hist(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return [{"timestamp": int(t.timestamp() * 1000), "openInterestValue": v}
            for t, v in zip(idx, values)]


def daily(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=idx)


class FlowContext(unittest.TestCase):
    RISING = [100, 104, 108, 112, 116, 120, 125, 130]      # +30% over 7 days
    UP = [10, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.5]    # +15%

    def test_new_longs_with_flat_funding_is_spot_led(self):
        f = m.flow_context(oi_hist(self.RISING), daily(self.UP), 11.5, 0.0, 7e9)
        self.assertEqual((f["quadrant"], f["flags"], f["oi_chg_pct"]),
                         ("NEW LONGS", ["SPOT-LED"], 30.0))

    def test_oi_above_daily_volume_multiple_is_flagged(self):
        f = m.flow_context(oi_hist(self.RISING), daily(self.UP), 11.5, None, 70.0)
        self.assertEqual(f["flags"], ["HIGH OI/VOL"])      # OI 130 vs $10/day volume

    def test_quiet_oi_gives_no_context(self):
        flat = [100, 100, 101, 100, 101, 100, 101, 102]
        self.assertIsNone(m.flow_context(oi_hist(flat), daily(self.UP), 11.5, 0.0, 7e9))

    def test_short_history_gives_no_context(self):
        self.assertIsNone(m.flow_context(oi_hist([100, 130]), daily(self.UP), 11.5, 0.0, 7e9))
        self.assertIsNone(m.flow_context(None, daily(self.UP), 11.5, 0.0, 7e9))


def full_sig(tf, side, **kw):
    base = dict(SIG, tf=tf, side=side, pair="ENS/USDT:USDT", base="ENS",
                current_price=100.5, weekly_volume=9e7, score=60)
    return dict(base, **kw)


class BuildRows(unittest.TestCase):
    def test_same_side_on_both_timeframes_is_one_row_4h_first(self):
        rows = m.build_rows([full_sig("1h", "bullish", score=80),
                             full_sig("4h", "bullish", score=70)])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["tfs"], r["score"], sorted(r["signals"]), r["primary_tf"]),
                         (["4h", "1h"], 80, ["1h", "4h"], "4h"))

    def test_opposite_sides_are_two_rows(self):
        rows = m.build_rows([full_sig("4h", "bullish"), full_sig("1h", "bearish")])
        self.assertEqual(sorted((r["side"], r["tfs"][0]) for r in rows),
                         [("bearish", "1h"), ("bullish", "4h")])

    def test_fresh_1h_is_primary_when_the_4h_signal_is_dead(self):
        r = m.build_rows([full_sig("4h", "bullish", status="invalidated"),
                          full_sig("1h", "bullish")])[0]
        self.assertEqual((r["primary_tf"], r["entry"]["verdict"], r["status"], r["tfs"]),
                         ("1h", "enter", "active", ["4h", "1h"]))

    def test_4h_is_primary_on_a_verdict_tie(self):
        r = m.build_rows([full_sig("4h", "bullish"), full_sig("1h", "bullish")])[0]
        self.assertEqual(r["primary_tf"], "4h")

    def test_wait_beats_skip(self):
        r = m.build_rows([full_sig("4h", "bullish", fresh=False),
                          full_sig("1h", "bullish", macd_cross=False)])[0]
        self.assertEqual((r["primary_tf"], r["entry"]["verdict"]), ("1h", "wait"))

    def test_context_and_event_are_carried_onto_the_row(self):
        rows = m.build_rows([full_sig("1h", "bearish")], mover={"flags": ["VOLUME SPIKE"]},
                            flow=None, context_error=None,
                            hours_to_event=3.0, event_title="FOMC Statement")
        r = rows[0]
        self.assertEqual((r["mover"], r["flow"]), ({"flags": ["VOLUME SPIKE"]}, None))
        self.assertIn("FOMC Statement is due in 3h", r["entry"]["reason"])


if __name__ == "__main__":
    unittest.main()
