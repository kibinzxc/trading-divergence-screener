"""Tests for the thin-coin filters and the cross-exchange merge.

Run with:  python -m unittest tests.test_venues -v
"""
import os
import re
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import LONG, wave_df  # noqa: E402

R = m.STRICTNESS["balanced"]


class ThinFilters(unittest.TestCase):
    def test_spread(self):
        self.assertFalse(m.spread_too_wide({"bid": 99.95, "ask": 100.05}))   # exactly 0.10%
        self.assertTrue(m.spread_too_wide({"bid": 99.9, "ask": 100.1}))      # 0.20%
        for unknown in (None, {}, {"bid": None, "ask": 100}, {"bid": 0, "ask": 1}):
            self.assertFalse(m.spread_too_wide(unknown), unknown)

    def test_atr_pct(self):
        self.assertFalse(m.atr_pct_out_of_range(0.6, 100))
        self.assertFalse(m.atr_pct_out_of_range(8.0, 100))
        self.assertTrue(m.atr_pct_out_of_range(0.5, 100))
        self.assertTrue(m.atr_pct_out_of_range(8.1, 100))
        self.assertFalse(m.atr_pct_out_of_range(float("nan"), 100))
        self.assertFalse(m.atr_pct_out_of_range(1.0, 0))

    def test_open_interest_reads_the_latest_point(self):
        old = {"timestamp": 1, "openInterestValue": 9e6}
        new = {"timestamp": 2, "openInterestAmount": 40_000}
        self.assertTrue(m.oi_too_thin([new, old], 100.0))          # 40k x $100 = $4M
        self.assertFalse(m.oi_too_thin([new, old], 200.0))         # $8M
        self.assertFalse(m.oi_too_thin(None, 100.0))
        self.assertFalse(m.oi_too_thin([{"timestamp": 3}], 100.0))  # unknown value


class Merge(unittest.TestCase):
    def test_binance_first_else_the_bigger_of_okx_and_bybit(self):
        perps = {"binance": ["BTC/USDT:USDT"],
                 "okx": ["BTC/USDT:USDT", "GRASS/USDT:USDT", "X/USDT:USDT"],
                 "bybit": ["BTC/USDT:USDT", "GRASS/USDT:USDT"]}
        vols = {("binance", "BTC/USDT:USDT"): 1.0, ("okx", "BTC/USDT:USDT"): 9.0,
                ("okx", "GRASS/USDT:USDT"): 2.0, ("bybit", "GRASS/USDT:USDT"): 5.0}
        listed, pick = m.coin_venues(perps, vols, "binance")
        self.assertEqual(sorted(pick), [("binance", "BTC/USDT:USDT"), ("bybit", "GRASS/USDT:USDT"),
                                        ("okx", "X/USDT:USDT")])
        self.assertEqual(listed["GRASS"], {"okx": "GRASS/USDT:USDT", "bybit": "GRASS/USDT:USDT"})

    def test_single_exchange_is_unchanged(self):
        listed, pick = m.coin_venues({"okx": ["A/USDT:USDT"]}, {}, "okx")
        self.assertEqual(pick, [("okx", "A/USDT:USDT")])

    def test_tags(self):
        self.assertEqual(m.venue_tags("binance", []), [])
        self.assertEqual(m.venue_tags("binance", ["bybit", "okx"]), ["ALSO ON OKX, BYBIT"])
        self.assertEqual(m.venue_tags("okx", ["bybit"]), ["NOT ON BINANCE", "ALSO ON BYBIT"])


class Confirmation(unittest.TestCase):
    def setUp(self):
        self.df = wave_df(LONG)            # bullish, signal bar 94
        self.sig = m.signals_for_tf(self.df, "1h", R, 100.5, "X/USDT:USDT", 9e7)[0]

    def shifted(self, hours):
        df = self.df.copy()
        df.index = df.index + pd.Timedelta(hours=hours)
        return df

    def test_same_side_within_one_bar_confirms(self):
        self.assertTrue(m.venue_confirms(self.sig, self.df, None, R))
        self.assertTrue(m.venue_confirms(self.sig, self.shifted(1), None, R))
        self.assertTrue(m.venue_confirms(self.sig, self.shifted(-1), None, R))

    def test_further_away_or_other_side_does_not(self):
        self.assertFalse(m.venue_confirms(self.sig, self.shifted(2), None, R))
        self.assertFalse(m.venue_confirms(dict(self.sig, side="bearish"), self.df, None, R))
        self.assertFalse(m.venue_confirms(self.sig, None, None, R))


class Page(unittest.TestCase):
    def test_all_is_the_default_exchange_and_its_tags_are_defined(self):
        html = m.app.test_client().get("/").get_data(as_text=True)
        self.assertRegex(html, r'<select id="exchange"><option value="all" selected>')
        tip = re.search(r"var TIP = \{(.*?)\n\};", html, re.S).group(1)
        for k in ("'ALSO ON':", "'NOT ON BINANCE':"):
            self.assertIn(k, tip, k)


if __name__ == "__main__":
    unittest.main()
