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


def bars(high, low, close):
    return pd.DataFrame({"high": high, "low": low, "close": close}, dtype=float)


class Outcome(unittest.TestCase):
    # entry 100, risk 10: the 0.10% round-trip fee is 0.1 / 10 = 0.01R off every result
    def test_stop_is_checked_before_the_target_inside_one_bar(self):
        df = bars([100, 140], [100, 80], [100, 100])
        self.assertEqual(m.trade_result(df, 0, "bullish", 100.0, 90.0, 2, 30),
                         {"r": -1.01, "bars": 1, "mae_r": 2.0})

    def test_each_target_is_its_own_exit(self):
        df = bars([100, 101, 100], [100, 79, 69], [100, 90, 70])
        self.assertEqual(m.trade_result(df, 0, "bearish", 100.0, 110.0, 3, 30),
                         {"r": 2.99, "bars": 2, "mae_r": 0.1})
        self.assertEqual(m.trade_result(df, 0, "bearish", 100.0, 110.0, 2, 30),
                         {"r": 1.99, "bars": 1, "mae_r": 0.1})

    def test_time_exit_marks_the_close_and_short_history_is_unresolved(self):
        df = bars([100, 105, 108], [100, 98, 99], [100, 104, 106])
        self.assertEqual(m.trade_result(df, 0, "bullish", 100.0, 90.0, 3, 2),
                         {"r": 0.59, "bars": 2, "mae_r": 0.2})
        self.assertIsNone(m.trade_result(df, 0, "bullish", 100.0, 90.0, 3, 5))

    def test_fee_uses_the_risk_floor_of_half_a_percent(self):
        df = bars([100, 101], [100, 99], [100, 100])
        # stop 0.1 away, risk floored at 0.5: fee 0.1 / 0.5 = 0.2R
        self.assertEqual(m.trade_result(df, 0, "bullish", 100.0, 99.9, 1, 30)["r"], -1.2)


class Replay(unittest.TestCase):
    def test_signal_is_taken_on_the_bar_it_first_exists(self):
        t = m.backtest_pair(frame(), "1h", RULES)
        self.assertEqual(len(t), 1)
        self.assertEqual((t[0]["side"], t[0]["tf"], t[0]["time"]),
                         ("bullish", "1h", str(wave_df(REPLAY).index[126])))
        # 27 bars left, short of the 48-bar time exit and nothing hit: unresolved
        self.assertEqual((t[0]["r1"], t[0]["r2"], t[0]["r3"], t[0]["bars"]), (None, None, None, None))
        # no HTF, OI or BTC here: the pattern points alone, rescaled (46.5 / 70)
        self.assertEqual((t[0]["score"], t[0]["grade"], t[0]["regime"]), (66, "B", "COUNTER TREND"))
        # 0.05% slippage against a long; bar 126 of replayed 80..152 is in the first 70%
        self.assertAlmostEqual(t[0]["entry"], wave_df(REPLAY)["close"].iloc[126] * 1.0005)
        self.assertTrue(t[0]["in_sample"])

    def test_outcome_follows_the_bars_after_entry(self):
        win = m.backtest_pair(frame(high=130.0), "1h", RULES)[0]
        self.assertEqual(win["bars"], 4)
        self.assertAlmostEqual(win["r3"] - win["r1"], 2)
        self.assertTrue(0.8 < win["r1"] < 1)                 # 1R less the fee
        loss = m.backtest_pair(frame(low=80.0), "1h", RULES)[0]
        self.assertEqual(loss["bars"], 4)
        self.assertTrue(loss["r1"] == loss["r2"] == loss["r3"] < -1)

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


def trade(day, r1, r2, bars=2, mae=1.0, score=60, in_sample=True):
    return {"time": f"2026-01-{day:02d}", "r1": r1, "r2": r2, "bars": bars, "mae_r": mae,
            "score": score, "in_sample": in_sample}


class Summary(unittest.TestCase):
    def thirty(self):
        # every third trade runs to 2R; the next one reaches 1R and is then stopped; the third is stopped
        return [trade(d, .99, 1.99, 6, .2) if d % 3 == 1 else
                trade(d, .99, -1.01) if d % 3 == 2 else
                trade(d, -1.01, -1.01) for d in range(1, 31)]

    def test_group_statistics_on_the_2r_policy(self):
        ts = self.thirty() + [trade(31, .99, None)]           # unresolved at 2R: left out
        self.assertEqual(m.group_stats(ts[::-1]), {
            "n": 30, "win1": 66.7, "win2": 33.3, "exp1": 0.32, "exp2": -0.01,
            "pf": 0.99, "bars": 2, "mae": 0.73, "streak": 2})

    def test_losing_streak_follows_time_not_list_order(self):
        ts = self.thirty()
        for d in (4, 7, 10):                                  # wins on days 1, 13, 16 ...
            ts[d - 1] = trade(d, -1.01, -1.01)
        self.assertEqual(m.group_stats(sorted(ts, key=lambda t: t["r2"]))["streak"], 11)

    def test_fewer_than_30_is_insufficient(self):
        self.assertEqual(m.group_stats(self.thirty()[1:]), {"n": 29})


class Tuning(unittest.TestCase):
    def trades(self, oos_r2):
        return ([trade(1, 0, .5, score=80) for _ in range(30)] +
                [trade(1, 0, -.5, score=60) for _ in range(30)] +
                [trade(1, 0, oos_r2, score=80, in_sample=False) for _ in range(5)])

    def test_lowest_in_sample_score_with_30_trades_at_plus_0_3r(self):
        c, why = m.tune_cutoff(self.trades(.3))               # 0.3 >= half of 0.5
        self.assertEqual(c, 80)
        self.assertIn("80", why)

    def test_rejected_when_out_of_sample_falls_below_half(self):
        c, why = m.tune_cutoff(self.trades(.2))
        self.assertIsNone(c)
        self.assertIn("70 / 55", why)

    def test_nothing_qualifies_in_sample(self):
        c, why = m.tune_cutoff(self.trades(.3)[30:])
        self.assertIsNone(c)
        self.assertIn("70 / 55", why)


if __name__ == "__main__":
    unittest.main()
