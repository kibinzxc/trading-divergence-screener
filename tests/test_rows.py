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

SIG = {"tf": "1h", "status": "active", "fresh": True, "bars_ago": 2, "r_reached": 0.0,
       "pct_chg": 0.4, "signal_price": 100.0, "stop_price": 97.0, "risk_pct": 3.0,
       "grade": "A", "score": 74, "regime": "WITH TREND",
       "forming": False, "trigger_first": False}


def sig(**kw):
    return dict(SIG, **kw)


class EntryVerdict(unittest.TestCase):
    def verdict(self, s, *a):
        return m.entry_verdict(s, *a)["verdict"]

    def test_confirmed_yes_when_every_check_passes(self):
        v = m.entry_verdict(sig())
        self.assertEqual((v["verdict"], v["level"]), ("enter", 97.0))
        self.assertTrue(v["reason"].startswith("Yes (confirmed)."))
        self.assertIn("Risk 3.0%", v["reason"])

    def test_invalidated_is_no(self):
        v = m.entry_verdict(sig(status="invalidated"))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("hit the stop", v["reason"])

    def test_ran_is_no(self):
        v = m.entry_verdict(sig(status="ran", r_reached=2.5))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("2.5x the risk", v["reason"])

    def test_fresh_windows_are_48h_on_4h_and_12h_on_1h(self):
        self.assertEqual((m.TF_CFG["4h"]["fresh_h"], m.TF_CFG["1h"]["fresh_h"]), (48, 12))

    def test_old_signal_with_price_still_at_entry_is_watch_only(self):
        v = m.entry_verdict(sig(fresh=False, bars_ago=20))
        self.assertEqual((v["verdict"], v["level"]), ("wait", None))
        self.assertTrue(v["reason"].startswith("Not yet. Watch only:"))
        self.assertIn("20 completed 1H bars (20h) have closed after the signal bar", v["reason"])
        self.assertIn("past the 12 bars (12h) for a fresh entry on 1H", v["reason"])
        v = m.entry_verdict(sig(tf="4h", fresh=False, bars_ago=13))
        self.assertIn("13 completed 4H bars (52h)", v["reason"])
        self.assertIn("past the 12 bars (48h) for a fresh entry on 4H", v["reason"])

    def test_old_signal_never_becomes_yes(self):
        for kw in ({}, {"forming": True, "trigger_first": True}, {"grade": "B"}, {"regime": "COUNTER TREND", "grade": "A"}):
            self.assertEqual(self.verdict(sig(fresh=False, bars_ago=20, **kw)), "wait", kw)

    def test_old_signal_is_no_once_price_has_left_the_entry_or_the_grade_is_c(self):
        v = m.entry_verdict(sig(fresh=False, bars_ago=20, status="extended", pct_chg=3.1))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("20 completed 1H bars (20h) have closed after the signal bar", v["reason"])
        self.assertIn("limit on 1H is 12 bars, 12h", v["reason"])
        self.assertIn("moved away", v["reason"])
        self.assertEqual(self.verdict(sig(fresh=False, bars_ago=20, grade="C", score=41)), "skip")
        self.assertEqual(self.verdict(sig(fresh=False, bars_ago=20, status="invalidated")), "skip")
        self.assertEqual(self.verdict(sig(fresh=False, bars_ago=20, status="ran", r_reached=2.5)), "skip")

    def test_grade_c_is_no(self):
        v = m.entry_verdict(sig(grade="C", score=41))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("grade C", v["reason"])

    def test_counter_trend_needs_grade_a(self):
        v = m.entry_verdict(sig(regime="COUNTER TREND", grade="B"))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("Watch only", v["reason"])
        self.assertEqual(self.verdict(sig(regime="COUNTER TREND", grade="A")), "enter")

    def test_chop_keeps_the_normal_cutoff_for_a_reversal(self):
        self.assertEqual(self.verdict(sig(regime="CHOP", grade="B")), "enter")

    def test_extended_waits_for_a_pullback_to_the_signal_price(self):
        v = m.entry_verdict(sig(status="extended", pct_chg=3.1))
        self.assertEqual((v["verdict"], v["level"]), ("wait", 100.0))
        self.assertIn("3.1%", v["reason"])

    def test_blocker_inside_the_timeframe_window_is_wait_and_names_it(self):
        v = m.entry_verdict(sig(), (3.2, "CPI m/m"))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("CPI m/m is due in 3h", v["reason"])
        v = m.entry_verdict(sig(tf="4h"), (11.8, "FOMC Statement"))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("FOMC Statement is due in 12h", v["reason"])

    def test_blocker_that_printed_in_the_last_hour_is_wait(self):
        v = m.entry_verdict(sig(), (-0.5, "Non-Farm Employment Change"))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("Non-Farm Employment Change printed 30m ago", v["reason"])

    def test_blocker_outside_the_window_or_unknown_does_not_block(self):
        self.assertEqual(self.verdict(sig(), (4.5, "CPI m/m")), "enter")        # 1H waits 4h
        self.assertEqual(self.verdict(sig(tf="4h"), (4.5, "CPI m/m")), "wait")
        self.assertEqual(self.verdict(sig(tf="4h"), (12.5, "CPI m/m")), "enter")  # 4H waits 12h
        self.assertEqual(self.verdict(sig(), (-1.5, "CPI m/m")), "enter")
        self.assertEqual(self.verdict(sig(), None), "enter")

    def test_forming_bar_is_early_only_with_the_trigger(self):
        v = m.entry_verdict(sig(forming=True, trigger_first=True))
        self.assertEqual(v["verdict"], "enter")
        self.assertTrue(v["reason"].startswith("Yes (early)."))
        v = m.entry_verdict(sig(forming=True))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("still forming", v["reason"])

    def test_rules_apply_in_order(self):
        self.assertEqual(self.verdict(sig(status="invalidated"), (2.0, "FOMC")), "skip")
        self.assertIn("pull back", m.entry_verdict(sig(status="extended"), (2.0, "FOMC"))["reason"])
        self.assertEqual(self.verdict(sig(grade="C", regime="COUNTER TREND")), "skip")
        self.assertIn("Watch only", m.entry_verdict(
            sig(grade="B", regime="COUNTER TREND", status="extended"))["reason"])


def ev(ts, title, currency="USD", mover=True):
    return {"ts": ts, "title": title, "currency": currency, "mover": mover}


class MacroWindow(unittest.TestCase):
    NOW = dt.datetime(2026, 11, 19, 12, 0, tzinfo=dt.timezone.utc)

    def test_nearest_blocker_and_nearest_other_mover(self):
        events = [ev("2026-11-19T10:00:00Z", "CPI m/m"),                 # printed 2h ago: over
                  ev("2026-11-19T15:00:00Z", "Unemployment Claims", mover=False),
                  ev("2026-11-19T15:00:00Z", "CPI m/m", "CAD", mover=False),
                  ev("2026-11-19T20:00:00Z", "FOMC Meeting Minutes"),    # a mover, not a blocker
                  ev("2026-11-19T18:00:00Z", "BOJ Policy Rate", "JPY"),
                  ev("2026-11-20T13:30:00Z", "Core PCE Price Index m/m"),
                  ev("2026-11-21T19:00:00Z", "FOMC Statement")]
        self.assertEqual(m.macro_window(events, self.NOW),
                         ((25.5, "Core PCE Price Index m/m"), (6.0, "BOJ Policy Rate")))

    def test_a_blocker_keeps_blocking_for_an_hour_after_it_printed(self):
        events = [ev("2026-11-19T11:30:00Z", "Non-Farm Employment Change"),
                  ev("2026-11-19T14:00:00Z", "Federal Funds Rate")]
        self.assertEqual(m.macro_window(events, self.NOW)[0], (-0.5, "Non-Farm Employment Change"))

    def test_every_blocker_on_the_list(self):
        for t in ("CPI m/m", "Core CPI m/m", "Core PCE Price Index m/m", "PPI m/m", "Core PPI m/m",
                  "Non-Farm Employment Change", "Unemployment Rate", "Federal Funds Rate",
                  "FOMC Statement", "FOMC Press Conference", "Fed Chair Powell Speaks",
                  "Fed Chair Powell Testifies"):
            self.assertEqual(m.macro_window([ev("2026-11-19T14:00:00Z", t)], self.NOW),
                             ((2.0, t), None), t)
        for t in ("Average Hourly Earnings m/m", "Advance GDP q/q", "Retail Sales m/m",
                  "ISM Manufacturing PMI", "FOMC Economic Projections"):
            self.assertEqual(m.macro_window([ev("2026-11-19T14:00:00Z", t)], self.NOW),
                             (None, (2.0, t)), t)

    def test_other_movers_only_within_24h_and_none_when_nothing_is_scheduled(self):
        self.assertEqual(m.macro_window([ev("2026-11-20T13:00:00Z", "Main Refinancing Rate", "EUR")],
                                        self.NOW), (None, None))
        self.assertEqual(m.macro_window([], self.NOW), (None, None))


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

    def test_quiet_oi_keeps_the_number_but_no_tags(self):
        flat = [100, 100, 101, 100, 101, 100, 101, 102]
        self.assertEqual(m.flow_context(oi_hist(flat), daily(self.UP), 11.5, 0.0, 7e9),
                         {"quadrant": None, "flags": [], "oi_chg_pct": 2.0})

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
                         (["4h", "1h"], 70, ["1h", "4h"], "4h"))

    def test_opposite_sides_are_two_rows(self):
        rows = m.build_rows([full_sig("4h", "bullish"), full_sig("1h", "bearish")])
        self.assertEqual(sorted((r["side"], r["tfs"][0]) for r in rows),
                         [("bearish", "1h"), ("bullish", "4h")])

    def test_fresh_1h_is_primary_when_the_4h_signal_is_dead(self):
        r = m.build_rows([full_sig("4h", "bullish", status="invalidated"),
                          full_sig("1h", "bullish")])[0]
        self.assertEqual((r["primary_tf"], r["entry"]["verdict"], r["status"], r["tfs"]),
                         ("1h", "enter", "active", ["4h", "1h"]))

    def test_row_carries_each_timeframes_own_verdict(self):
        r = m.build_rows([full_sig("4h", "bullish", status="invalidated"), full_sig("1h", "bullish")])[0]
        self.assertEqual({tf: v["verdict"] for tf, v in r["verdicts"].items()}, {"4h": "skip", "1h": "enter"})
        self.assertIn("hit the stop", r["verdicts"]["4h"]["reason"])
        self.assertEqual(r["verdicts"][r["primary_tf"]], r["entry"])
        one = m.build_rows([full_sig("1h", "bearish")])[0]
        self.assertEqual(list(one["verdicts"]), ["1h"])

    def test_4h_is_primary_on_a_verdict_tie(self):
        r = m.build_rows([full_sig("4h", "bullish"), full_sig("1h", "bullish")])[0]
        self.assertEqual(r["primary_tf"], "4h")

    def test_wait_beats_skip(self):
        r = m.build_rows([full_sig("4h", "bullish", status="invalidated"),
                          full_sig("1h", "bullish", status="extended")])[0]
        self.assertEqual((r["primary_tf"], r["entry"]["verdict"]), ("1h", "wait"))

    def test_row_takes_grade_and_regime_from_the_primary_signal(self):
        r = m.build_rows([full_sig("4h", "bullish", status="invalidated", grade="A"),
                          full_sig("1h", "bullish", grade="B", score=61)])[0]
        self.assertEqual((r["primary_tf"], r["grade"], r["score"], r["regime"]),
                         ("1h", "B", 61, "WITH TREND"))

    def test_context_and_event_are_carried_onto_the_row(self):
        rows = m.build_rows([full_sig("1h", "bearish")], mover={"flags": ["VOLUME SPIKE"]},
                            flow=None, context_error=None,
                            blocker=(3.0, "FOMC Statement"), macro=(6.2, "BOJ Policy Rate"))
        r = rows[0]
        self.assertEqual((r["mover"], r["flow"]), ({"flags": ["VOLUME SPIKE"]}, None))
        self.assertIn("FOMC Statement is due in 3h", r["entry"]["reason"])
        self.assertEqual(r["macro"], "MACRO BOJ Policy Rate in 6h")
        self.assertIsNone(m.build_rows([full_sig("1h", "bearish")])[0]["macro"])

    def test_a_blocker_6h_out_holds_the_4h_signal_but_not_the_1h(self):
        r = m.build_rows([full_sig("4h", "bullish"), full_sig("1h", "bullish")],
                         blocker=(6.0, "CPI m/m"))[0]
        self.assertEqual((r["primary_tf"], r["entry"]["verdict"]), ("1h", "enter"))
        self.assertEqual(r["macro"], "MACRO CPI m/m in 6h")      # due, and the answer does not say so

    def test_a_blocker_due_within_24h_that_the_answer_does_not_name_is_the_macro_tag(self):
        def tag(sigs, blocker, macro=None):
            return m.build_rows(sigs, blocker=blocker, macro=macro)[0]["macro"]
        h4 = [full_sig("4h", "bullish")]
        self.assertEqual(tag(h4, (13.0, "CPI m/m")), "MACRO CPI m/m in 13h")
        self.assertIsNone(tag(h4, (3.0, "CPI m/m")))                       # the answer names it
        self.assertIsNone(tag(h4, (25.0, "CPI m/m")))                      # beyond 24h
        self.assertIsNone(tag(h4, (-0.5, "CPI m/m")))                      # already printed
        self.assertEqual(tag([full_sig("4h", "bullish", grade="C")], (3.0, "CPI m/m")),
                         "MACRO CPI m/m in 3h")                            # No for another reason
        self.assertEqual(tag(h4, (13.0, "CPI m/m"), (2.0, "BOJ Policy Rate")),
                         "MACRO BOJ Policy Rate in 2h")                    # the sooner one
        self.assertEqual(tag(h4, (13.0, "CPI m/m"), (20.0, "BOJ Policy Rate")),
                         "MACRO CPI m/m in 13h")


if __name__ == "__main__":
    unittest.main()
