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
        self.assertIn("risk 3.0%", v["reason"])

    def test_invalidated_is_skip(self):
        v = m.entry_verdict(sig(status="invalidated"))
        self.assertEqual((v["verdict"], v["reason"]), ("skip", "Stop was hit"))

    def test_ran_is_skip(self):
        v = m.entry_verdict(sig(status="ran", r_reached=2.5))
        self.assertEqual((v["verdict"], v["reason"]), ("skip", "Already ran 2.5R"))

    def test_not_fresh_is_skip(self):
        v = m.entry_verdict(sig(fresh=False, bars_ago=20))
        self.assertEqual((v["verdict"], v["reason"]), ("skip", "Too old (20 bars)"))

    def test_extended_waits_for_a_pullback_to_the_signal_price(self):
        v = m.entry_verdict(sig(status="extended", pct_chg=3.1))
        self.assertEqual((v["verdict"], v["level"]), ("wait", 100.0))
        self.assertIn("3.1%", v["reason"])

    def test_no_macd_cross_is_wait(self):
        self.assertEqual(self.verdict(sig(macd_cross=False)), "wait")

    def test_event_inside_24h_is_wait_and_names_the_event(self):
        v = m.entry_verdict(sig(), 14.2, "CPI m/m")
        self.assertEqual((v["verdict"], v["reason"]), ("wait", "CPI m/m in 14h"))

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


if __name__ == "__main__":
    unittest.main()
