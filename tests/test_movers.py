"""Tests for the movers thin band.

Run with:  python -m unittest tests.test_movers -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
