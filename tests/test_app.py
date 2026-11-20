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
                    "ts": "2026-11-19T00:00:00+00:00", "meta": {"tfs": ["4h", "1h"], "v": 3}})
        m.load_state()
        self.assertEqual(m._state["results"], [{"pair": "X", "tfs": ["4h"]}])
        m.save_state()
        with open(m.STATE_FILE) as f:
            saved = json.load(f)
        self.assertEqual(sorted(saved), ["meta", "results", "ts"])
        self.assertEqual(saved["meta"]["v"], 3)

    def test_state_from_before_the_v3_score_is_ignored(self):
        self.write({"results": [{"pair": "X", "tfs": ["4h"]}],
                    "ts": "2026-11-19T00:00:00+00:00", "meta": {"tfs": ["4h", "1h"], "v": 2}})
        m.load_state()
        self.assertEqual(m._state["results"], [])


import re  # noqa: E402


class BriefMeta(unittest.TestCase):
    NOW = m.datetime.datetime(2026, 11, 20, 12, 0, tzinfo=m.datetime.timezone.utc)
    MS = int(NOW.timestamp() * 1000)

    def btc(self, rsi):
        import pandas as pd
        return pd.DataFrame({"rsi": [50.0, rsi], "close": [100.0, 110.0], "ema200": [100.0, 100.0]})

    def test_btc_is_the_last_completed_bar(self):
        d = m.brief_meta(self.btc(60.0), (None, None), self.NOW)
        self.assertEqual(d["btc"], {"state": "BULL", "rsi": 60.0, "close": 110.0, "ema200": 100.0})
        self.assertEqual(m.brief_meta(self.btc(50.0), (None, None), self.NOW)["btc"]["state"], "CHOP")

    def test_calendar_entries_become_absolute_times(self):
        d = m.brief_meta(None, ((6.0, "CPI m/m"), (-0.5, "BOJ Policy Rate")), self.NOW)
        self.assertEqual(d, {"btc": None,
                             "blocker": {"ts_ms": self.MS + 6 * 3_600_000, "title": "CPI m/m"},
                             "macro": {"ts_ms": self.MS - 1_800_000, "title": "BOJ Policy Rate"}})
        json.dumps(d)   # saved with the state

    def test_nothing_due_is_null_and_no_calendar_leaves_the_keys_out(self):
        self.assertEqual(m.brief_meta(None, (None, None), self.NOW),
                         {"btc": None, "blocker": None, "macro": None})
        self.assertEqual(m.brief_meta(None, None, self.NOW), {"btc": None})


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

    def test_level_labels_keep_plain_digits(self):
        self.assertEqual([m._plain(v) for v in (65432.1, 1.25, 0.0000098)],
                         ["65432.1", "1.25", "0.0000098"])

    def test_route_serves_the_chart_of_a_scanned_row(self):
        r = self.c.get("/chart", query_string={"pair": "ENS/USDT:USDT", "side": "bullish", "tf": "1h"})
        self.assertEqual((r.status_code, r.mimetype, r.data[:4]), (200, "image/png", b"\x89PNG"))

    def test_unknown_row_is_404(self):
        r = self.c.get("/chart", query_string={"pair": "NOPE/USDT:USDT", "side": "bullish", "tf": "1h"})
        self.assertEqual(r.status_code, 404)


class Page(unittest.TestCase):
    def setUp(self):
        self.html = m.app.test_client().get("/").get_data(as_text=True)

    def test_five_pages(self):
        self.assertEqual(re.findall(r'class="nav-btn[^"]*" data-pg="(\w+)"', self.html),
                         ["scan", "movers", "calendar", "tags", "learn"])
        self.assertEqual(sorted(re.findall(r'<div id="pg-(\w+)"', self.html)),
                         ["calendar", "learn", "movers", "scan", "tags"])

    def test_oi_column_and_tags_page(self):
        for want in ('data-sort="oi"', "function oiCell", 'id="tagsBody"', 'colspan="7"'):
            self.assertIn(want, self.html, want)
        # every tag the Tags page lists has a definition
        groups = re.search(r"var TAG_GROUPS = \[(.*?)\n\];", self.html, re.S).group(1)
        tip = re.search(r"var TIP = \{(.*?)\n\};", self.html, re.S).group(1)
        for k in ("'CHOP'", "'PARTIAL DATA'"):
            self.assertIn(k, groups, k)
        for line in groups.splitlines():
            for k in re.findall(r"'([^']+)'", line.split(", [", 1)[-1]):
                self.assertIn("'" + k + "':", tip, k)

    def test_no_unfilled_template_placeholders(self):
        self.assertEqual(re.findall(r"__[A-Z][A-Z_]+__", self.html), [])

    def test_removed_features_leave_no_code_behind(self):
        for gone in ("startWatch", "rtGo", "loadDash",
                     'data-val="loose"', 'data-val="1d"', "/watchlist"):
            self.assertNotIn(gone, self.html, gone)

    def test_new_table_and_filters(self):
        for want in ('data-f="enter"', 'data-f="both"', "function evCell", "function tagChips",
                     'id="scan-age"', 'id="dash-cal"',
                     "function gradeCell", "/chart?pair=", "'W-PATTERN'", "'LTF TRIGGER'"):
            self.assertIn(want, self.html, want)

    def test_seven_columns(self):
        head = re.search(r'<table id="resT">\s*<thead>(.*?)</thead>', self.html, re.S).group(1)
        self.assertEqual(re.findall(r"<th[^>]*>([^<]+)</th>", head),
                         ["Coin", "Side and TF", "Grade", "Answer", "Entry", "Stop", "7d OI"])
        for gone in ('colspan="11"', "function ctxCell", 'data-sort="age"'):
            self.assertNotIn(gone, self.html, gone)

    def test_brief_and_fold(self):
        for want in ('id="briefH"', 'id="briefState"', 'id="picks"', "function renderBrief",
                     "Worth a look", "Closest to qualifying", "No trade is a fine answer",
                     "the backtest has not shown an edge yet",
                     '<button class="fold" id="fold"', "that did not pass, newest first",
                     # rows open with click, Enter or Space
                     'tabindex="0"', "e.key === 'Enter' || e.key === ' '",
                     # the brief copes with a scan saved before its meta fields existed
                     "m.blocker === null", "b === null"):
            self.assertIn(want, self.html, want)

    def test_settings_live_in_a_closed_panel_on_the_scan_page(self):
        scan = re.search(r'<div id="pg-scan".*?<div id="pg-movers"', self.html, re.S).group(0)
        panel = re.search(r'<details class="set" id="scanSet"><summary>Scan settings</summary>(.*?)</details>',
                          scan, re.S).group(1)
        for cid in ("exchange", "minVol", "uniBtn", "uniOut", "strG", "strHint", "maxPairs", "spdG",
                    "symOverride"):
            self.assertIn('id="%s"' % cid, panel, cid)
        for cid in ("runBtn", "pgSec", "pbFill", "pbLbl", "lc", "sl", "scan-audit"):
            self.assertIn('id="%s"' % cid, scan, cid)
        self.assertNotIn("<aside", self.html)

    def test_the_look(self):
        for token in ("--bg:#0C111B", "--sf:#121926", "--sf2:#182132", "--line:#222D40", "--tx:#DCE3EE",
                      "--tx2:#9AA8BD", "--tx3:#6E7D94", "--acc:#8FB4E8", "--yes:#6FCF9F",
                      "--wait:#E2B866", "--no:#8E9BB0", "--short:#E58C98",
                      "font-size:15px", ":focus-visible", "prefers-reduced-motion"):
            self.assertIn(token, self.html, token)
        # nothing below 12px (the Method drawings are SVG user units, scaled up on screen)
        css = "\n".join(l for l in self.html.splitlines() if not l.startswith(".lp-"))
        small = [v for v in re.findall(r"font-size:\s*([\d.]+)px", css) if float(v) < 12]
        self.assertEqual(small, [])
        # the old palette is gone
        for gone in ("var(--bull", "var(--bear", "var(--fresh", "var(--bd", "var(--sf3"):
            self.assertNotIn(gone, self.html, gone)

    def test_pivot_era_ui_is_gone(self):
        for gone in ("macd_cross", "volCell", "FROM CURRENT PRICE", "MACD pivot must align"):
            self.assertNotIn(gone, self.html, gone)


if __name__ == "__main__":
    unittest.main()
