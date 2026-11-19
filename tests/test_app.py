"""Tests for routes and saved state.

Run with:  python -m unittest tests.test_app -v
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import LONG, wave_df  # noqa: E402


class Routes(unittest.TestCase):
    def setUp(self):
        self.c = m.app.test_client()

    def test_removed_scans_are_gone(self):
        for path in ("/watchlist", "/watch_signals"):
            self.assertEqual(self.c.get(path).status_code, 404, path)

    def test_usdt_perps_keeps_only_active_crypto_linear_perps(self):
        mkts = {"ENS/USDT:USDT":  {"active": True,  "base": "ENS"},
                "USDC/USDT:USDT": {"active": True,  "base": "USDC"},
                "TSLA/USDT:USDT": {"active": True,  "base": "TSLA"},
                "OLD/USDT:USDT":  {"active": False, "base": "OLD"},
                "BTC/USD:BTC":    {"active": True,  "base": "BTC"}}
        self.assertEqual(m.usdt_perps(mkts), ["ENS/USDT:USDT"])

    def test_signals_returns_rows(self):
        m._state.update({"results": [{"pair": "ENS/USDT:USDT"}], "ts": None,
                         "meta": {"tfs": ["4h", "1h"]}})
        try:
            d = self.c.get("/signals").get_json()
            self.assertEqual((d["rows"], d["meta"]["tfs"], d["ts"]),
                             ([{"pair": "ENS/USDT:USDT"}], ["4h", "1h"], None))
        finally:
            m._state.update({"results": [], "ts": None, "meta": {}})


class SavedState(unittest.TestCase):
    def setUp(self):
        self._orig = m.STATE_FILE
        m.STATE_FILE = os.path.join(tempfile.mkdtemp(), "scan_state.json")
        m._state.update({"results": [], "ts": None, "meta": {}})

    def tearDown(self):
        m.STATE_FILE = self._orig
        m._state.update({"results": [], "ts": None, "meta": {}})

    def write(self, data):
        with open(m.STATE_FILE, "w") as f:
            json.dump(data, f)

    def test_state_from_before_wave_detection_is_ignored(self):
        self.write({"results": [{"pair": "X", "tfs": ["4h"]}],
                    "ts": "2026-11-01T00:00:00+00:00", "meta": {"tfs": ["4h", "1h"]}})
        m.load_state()
        self.assertEqual((m._state["results"], m._state["ts"]), ([], None))

    def test_row_state_round_trips(self):
        self.write({"results": [{"pair": "X", "tfs": ["4h"]}],
                    "ts": "2026-11-19T00:00:00+00:00", "meta": {"tfs": ["4h", "1h"], "v": 2}})
        m.load_state()
        self.assertEqual(m._state["results"], [{"pair": "X", "tfs": ["4h"]}])
        m.save_state()
        with open(m.STATE_FILE) as f:
            self.assertEqual(sorted(json.load(f)), ["meta", "results", "ts"])


import re  # noqa: E402


class Chart(unittest.TestCase):
    KEY = "ENS/USDT:USDT|bullish|1h"

    def setUp(self):
        s = m.signals_for_tf(wave_df(LONG), "1h", m.STRICTNESS["balanced"],
                             100.5, "ENS/USDT:USDT", 9e7)[0]
        self.hit = (s.pop("_df"), s.pop("_d"), s)
        m._charts[self.KEY] = self.hit
        self.c = m.app.test_client()

    def tearDown(self):
        m._charts.clear()

    def test_render_is_a_png(self):
        self.assertEqual(m.render_chart(*self.hit)[:4], b"\x89PNG")

    def test_route_serves_the_chart_of_a_scanned_row(self):
        r = self.c.get("/chart", query_string={"pair": "ENS/USDT:USDT", "side": "bullish", "tf": "1h"})
        self.assertEqual((r.status_code, r.mimetype, r.data[:4]), (200, "image/png", b"\x89PNG"))

    def test_unknown_row_is_404(self):
        r = self.c.get("/chart", query_string={"pair": "NOPE/USDT:USDT", "side": "bullish", "tf": "1h"})
        self.assertEqual(r.status_code, 404)


class Page(unittest.TestCase):
    def setUp(self):
        self.html = m.app.test_client().get("/").get_data(as_text=True)

    def test_four_pages(self):
        self.assertEqual(re.findall(r'class="nav-btn[^"]*" data-pg="(\w+)"', self.html),
                         ["scan", "movers", "calendar", "learn"])
        self.assertEqual(sorted(re.findall(r'<div id="pg-(\w+)"', self.html)),
                         ["calendar", "learn", "movers", "scan"])

    def test_no_unfilled_template_placeholders(self):
        self.assertEqual(re.findall(r"__[A-Z][A-Z_]+__", self.html), [])

    def test_removed_features_leave_no_code_behind(self):
        for gone in ("startWatch", "rtGo", "loadDash",
                     'data-val="loose"', 'data-val="1d"', "/watchlist"):
            self.assertNotIn(gone, self.html, gone)

    def test_new_table_and_filters(self):
        for want in ('data-f="enter"', 'data-f="both"', "function evCell", "function ctxCell",
                     'id="scan-age"', 'id="dash-cal"',
                     "function gradeCell", "/chart?pair=", "'W-PATTERN'", "'LTF TRIGGER'"):
            self.assertIn(want, self.html, want)

    def test_pivot_era_ui_is_gone(self):
        for gone in ("macd_cross", "volCell", "FROM CURRENT PRICE", "MACD pivot must align"):
            self.assertNotIn(gone, self.html, gone)


if __name__ == "__main__":
    unittest.main()
