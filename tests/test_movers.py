"""Tests for the movers classification used for context badges.

Run with:  python -m unittest tests.test_movers -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402


class ClassifyMover(unittest.TestCase):
    BASE = {"chg_1d_pct": 0.0, "to_r_hi_pct": 10.0, "to_r_lo_pct": 10.0}

    def row(self, **kw):
        return dict(self.BASE, **kw)

    def test_one_trigger_qualifies(self):
        self.assertTrue(m.classify_mover(self.row(vol_mult=3.0, range_mult=1.0))[0])
        self.assertTrue(m.classify_mover(self.row(vol_mult=1.0, range_mult=1.0, chg_1d_pct=8.0))[0])

    def test_flags_name_what_fired(self):
        flags = m.classify_mover(self.row(vol_mult=3.0, range_mult=2.0, to_r_hi_pct=1.0))[1]
        self.assertEqual(flags, ["VOLUME SPIKE", "RANGE EXPANSION", "AT RANGE HIGH"])

    def test_thin_coin_needs_volume_and_range_together(self):
        self.assertFalse(m.classify_mover(self.row(vol_mult=3.0, range_mult=1.0), thin=True)[0])
        self.assertTrue(m.classify_mover(self.row(vol_mult=3.0, range_mult=2.0), thin=True)[0])


class MoverSignals(unittest.TestCase):
    def test_returns_the_cached_movers_scan(self):
        d = m.app.test_client().get("/mover_signals").get_json()
        self.assertEqual(sorted(d), ["meta", "rows", "ts"])


if __name__ == "__main__":
    unittest.main()
