"""Tests for the bar-by-bar replay.

Run with:  python -m unittest tests.test_backtest -v
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import W, wave_df  # noqa: E402

RULES = m.STRICTNESS["balanced"]
# 154 bars. Wave B is closed by the cross at bar 124; the gap clears the
# whipsaw filter two bars later, so the signal first exists at bar 126.
REPLAY = [(90, 1, 0)] + W[1:4] + [(30, 1, 0)]


def frame(**after_130):
    df = wave_df(REPLAY)
    for col, val in after_130.items():
        df.iloc[130:, df.columns.get_loc(col)] = val
    return df


class Outcome(unittest.TestCase):
    def test_stop_is_checked_before_the_target_inside_one_bar(self):
        df = pd.DataFrame({"high": [100, 140.0], "low": [100, 80.0]})
        self.assertEqual(m.trade_outcome(df, 0, "bullish", 100.0, 90.0),
                         {"hit": 0, "stopped": True, "bars": 1, "mae_r": 2.0})

    def test_short_reaching_3r(self):
        df = pd.DataFrame({"high": [100, 101, 100.0], "low": [100, 79, 69.0]})
        self.assertEqual(m.trade_outcome(df, 0, "bearish", 100.0, 110.0),
                         {"hit": 3, "stopped": False, "bars": 2, "mae_r": 0.1})


class Replay(unittest.TestCase):
    def test_signal_is_taken_on_the_bar_it_first_exists(self):
        t = m.backtest_pair(frame(), "1h", RULES)
        self.assertEqual(len(t), 1)
        self.assertEqual((t[0]["side"], t[0]["tf"], t[0]["time"]),
                         ("bullish", "1h", str(wave_df(REPLAY).index[126])))
        self.assertEqual((t[0]["hit"], t[0]["stopped"], t[0]["bars"]), (0, False, None))
        # no HTF, OI or BTC here: the pattern points alone, rescaled (46.5 / 70)
        self.assertEqual((t[0]["score"], t[0]["grade"], t[0]["regime"]), (66, "B", "COUNTER TREND"))

    def test_outcome_follows_the_bars_after_entry(self):
        win = m.backtest_pair(frame(high=130.0), "1h", RULES)[0]
        self.assertEqual((win["hit"], win["stopped"], win["bars"]), (3, False, 4))
        loss = m.backtest_pair(frame(low=80.0), "1h", RULES)[0]
        self.assertEqual((loss["hit"], loss["stopped"], loss["bars"]), (0, True, 4))

    def test_no_lookahead_what_happens_later_cannot_change_the_signal(self):
        keys = ("time", "side", "score", "grade", "regime")
        a = m.backtest_pair(frame(), "1h", RULES)[0]
        b = m.backtest_pair(frame(high=130.0, low=80.0), "1h", RULES)[0]
        self.assertEqual([a[k] for k in keys], [b[k] for k in keys])


class SameMomentAsLive(unittest.TestCase):
    def test_context_is_judged_at_the_signal_bar_close_not_the_first_seen_bar(self):
        df = frame()
        # signal bar 124 closes 05:00; first seen at bar 126, whose close is 07:00.
        # The 4H bar opening 02:00 completes at 06:00: between the two moments.
        hi = pd.date_range("2026-01-04 18:00", periods=20, freq="4h")
        htf = pd.DataFrame({"macd": [float(i) if t < pd.Timestamp("2026-01-06 02:00") else -100.0
                                     for i, t in enumerate(hi)]}, index=hi)
        # BTC 4H is read 4h before the close: 01:00 vs 03:00. BULL up to 01:00, CHOP after.
        bi = pd.date_range("2026-01-05", periods=72, freq="h")
        btc = pd.DataFrame({"rsi": [60.0 if t <= pd.Timestamp("2026-01-06 01:00") else 40.0
                                    for t in bi], "close": 110.0, "ema200": 100.0}, index=bi)
        sig = m.signals_for_tf(df, "1h", RULES, 100.5, "X/USDT:USDT", 9e7)[0]
        m.add_context(sig, htf, None, None, None, btc)
        t = m.backtest_pair(df, "1h", RULES, htf, btc)[0]
        self.assertEqual((sig["htf_ok"], sig["regime"]), (True, "WITH TREND"))   # the moments differ
        self.assertEqual((t["score"], t["grade"], t["regime"]),
                         (sig["score"], sig["grade"], sig["regime"]))


class Summary(unittest.TestCase):
    def test_win_rate_and_average_r_per_group(self):
        trades = [{"grade": "A", "hit": 3, "stopped": False},
                  {"grade": "A", "hit": 1, "stopped": True},
                  {"grade": "A", "hit": 0, "stopped": False},      # still open: in n, not in the rates
                  {"grade": "B", "hit": 0, "stopped": True}]
        self.assertEqual(m.summarise(trades, "grade"), [
            {"grade": "A", "n": 3, "win1": 100.0, "avg_r1": 1.0,
             "win2": 50.0, "avg_r2": 0.5, "win3": 50.0, "avg_r3": 1.0},
            {"grade": "B", "n": 1, "win1": 0.0, "avg_r1": -1.0,
             "win2": 0.0, "avg_r2": -1.0, "win3": 0.0, "avg_r3": -1.0}])


if __name__ == "__main__":
    unittest.main()
