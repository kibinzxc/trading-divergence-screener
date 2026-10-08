"""Tests for the movers thin band and the mentor log.

Run with:  python -m unittest tests.test_mentor -v
"""
import datetime as dt
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402


def daily_df(start, n, open_=10.0, high=11.0, low=9.0, close=10.0, volume=1000.0):
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


class ClassifyMoverThin(unittest.TestCase):
    BASE = {"chg_1d_pct": 0.0, "to_r_hi_pct": 10.0, "to_r_lo_pct": 10.0}

    def row(self, **kw):
        return dict(self.BASE, **kw)

    def test_one_trigger_qualifies_a_liquid_pair(self):
        self.assertTrue(m.classify_mover(self.row(vol_mult=3.0, range_mult=1.0))[0])
        self.assertTrue(m.classify_mover(self.row(vol_mult=1.0, range_mult=1.0, chg_1d_pct=8.0))[0])

    def test_thin_pair_needs_volume_and_range_together(self):
        self.assertFalse(m.classify_mover(self.row(vol_mult=7.0, range_mult=1.0), thin=True)[0])
        self.assertFalse(m.classify_mover(self.row(vol_mult=1.0, range_mult=2.0), thin=True)[0])
        self.assertFalse(m.classify_mover(self.row(vol_mult=1.0, range_mult=1.0, chg_1d_pct=12.0),
                                          thin=True)[0])
        self.assertTrue(m.classify_mover(self.row(vol_mult=2.5, range_mult=1.6), thin=True)[0])


class ParseTicker(unittest.TestCase):
    BASES = {"ICP", "1INCH", "VELODROME", "MON", "XU"}

    def test_tradingview_perp_notation(self):
        self.assertEqual(m.parse_ticker("ICPUSDT.P", self.BASES), "ICP")
        self.assertEqual(m.parse_ticker(" 1inchusdt.p ", self.BASES), "1INCH")

    def test_bare_base_and_ccxt_symbol(self):
        self.assertEqual(m.parse_ticker("MON", self.BASES), "MON")
        self.assertEqual(m.parse_ticker("MON/USDT:USDT", self.BASES), "MON")

    def test_poster_truncating_a_long_name(self):
        self.assertEqual(m.parse_ticker("VELODROMEU", self.BASES), "VELODROME")

    def test_base_ending_in_u_beats_a_truncation_guess(self):
        self.assertEqual(m.parse_ticker("XU", self.BASES | {"X"}), "XU")

    def test_unknown_is_none(self):
        self.assertIsNone(m.parse_ticker("NOPEUSDT.P", self.BASES))
        self.assertIsNone(m.parse_ticker("", self.BASES))


class LiquidityTier(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(m.liquidity_tier(m.DEFAULT_MIN_WEEKLY_VOL), "liquid")
        self.assertEqual(m.liquidity_tier(m.DEFAULT_MIN_WEEKLY_VOL - 1), "thin")
        self.assertEqual(m.liquidity_tier(m.MOVER_THIN_MIN_WEEKLY_VOL), "thin")
        self.assertEqual(m.liquidity_tier(m.MOVER_THIN_MIN_WEEKLY_VOL - 1), "micro")
        self.assertIsNone(m.liquidity_tier(None))


class MentorOutcome(unittest.TestCase):
    POST = dt.date(2026, 10, 5)

    def make(self, fwd_rows):
        before = daily_df("2026-09-05", 30)          # 30 bars ending 2026-10-04, TR = 2.0
        rows = []
        for o, h, l, c in fwd_rows:
            rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 1000.0})
        idx = pd.date_range("2026-10-05", periods=len(rows), freq="D")
        return pd.concat([before, pd.DataFrame(rows, index=idx)])

    def test_final_window(self):
        r = m.mentor_outcome(self.make([(10, 12, 9.5, 11), (11, 13, 10, 12.5)]), self.POST)
        self.assertEqual(r["status"], "final")
        self.assertEqual(r["chg_24h"], 10.0)
        self.assertEqual(r["chg_48h"], 25.0)
        self.assertEqual(r["max_up"], 30.0)
        self.assertEqual(r["max_dn"], -5.0)
        self.assertEqual(r["range_atr"], 1.75)                # (13 - 9.5) / 2.0

    def test_liquidity_measured_on_the_week_before_the_post(self):
        # A huge volume day ON the post date must not change the tier.
        df = self.make([(10, 12, 9.5, 11), (11, 13, 10, 12.5)])
        df.loc[pd.Timestamp("2026-10-05"), "volume"] = 1e12
        r = m.mentor_outcome(df, self.POST, contract_size=2.0)
        self.assertEqual(r["wk_vol"], 7 * 1000.0 * 2.0 * 10.0)
        self.assertEqual(r["tier"], "micro")

    def test_partial_and_pending(self):
        self.assertEqual(m.mentor_outcome(self.make([(10, 12, 9.5, 11)]), self.POST)["status"],
                         "partial")
        r = m.mentor_outcome(self.make([]), self.POST)
        self.assertEqual(r["status"], "pending")
        self.assertIn("tier", r)

    def test_no_data(self):
        self.assertEqual(m.mentor_outcome(None, self.POST)["status"], "no_data")


class MentorSummary(unittest.TestCase):
    def test_only_final_picks_count_and_btc_is_kept_apart(self):
        fin = lambda tier, up, dn, c48: {"status": "final", "tier": tier, "max_up": up,
                                         "max_dn": dn, "chg_48h": c48, "range_atr": 1.0}
        posts = [{"results": {
            "AAA": fin("thin", 4.0, -2.0, -3.0),
            "BBB": fin("thin", 2.0, -6.0, 1.0),
            "CCC": dict(fin("thin", 50.0, -50.0, 50.0), status="partial"),
            "DDD": fin("liquid", 1.0, -1.0, 0.5),
            "BTC": fin("liquid", 0.5, -0.5, 0.1),
        }}]
        s = m.mentor_summary(posts)
        self.assertEqual(s["thin"]["n"], 2)
        self.assertEqual(s["thin"]["max_up"], 3.0)
        self.assertEqual(s["thin"]["abs_chg_48h"], 2.0)
        self.assertEqual(s["thin"]["range_pct"], 7.0)
        self.assertEqual(s["liquid"]["n"], 1)
        self.assertEqual(s["btc"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
