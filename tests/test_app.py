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


class Routes(unittest.TestCase):
    def setUp(self):
        self.c = m.app.test_client()

    def test_removed_scans_are_gone(self):
        for path in ("/watchlist", "/watch_signals"):
            self.assertEqual(self.c.get(path).status_code, 404, path)

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

    def test_old_flat_signal_state_is_ignored(self):
        self.write({"results": [{"pair": "X", "side": "bullish"}],
                    "ts": "2026-11-01T00:00:00+00:00", "meta": {"tf": "1d"}})
        m.load_state()
        self.assertEqual((m._state["results"], m._state["ts"]), ([], None))

    def test_row_state_round_trips(self):
        self.write({"results": [{"pair": "X", "tfs": ["4h"]}],
                    "ts": "2026-11-19T00:00:00+00:00", "meta": {"tfs": ["4h", "1h"]}})
        m.load_state()
        self.assertEqual(m._state["results"], [{"pair": "X", "tfs": ["4h"]}])
        m.save_state()
        with open(m.STATE_FILE) as f:
            self.assertEqual(sorted(json.load(f)), ["meta", "results", "ts"])


import re  # noqa: E402


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
                     'id="scan-age"', 'id="dash-cal"'):
            self.assertIn(want, self.html, want)


if __name__ == "__main__":
    unittest.main()
