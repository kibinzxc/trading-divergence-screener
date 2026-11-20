"""Tests for routes and saved state.

Run with:  python -m unittest tests.test_app -v
"""
import json
import os
import shutil
import subprocess
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


class OldSignals(unittest.TestCase):
    def test_cutoff_counts_completed_bars_since_the_signal_bar(self):
        self.assertEqual(m.MAX_SIGNAL_AGE_H, 72)
        for tf, bars, old in (("4h", 18, False), ("4h", 19, True), ("1h", 72, False), ("1h", 73, True),
                              ("1h", 0, False)):
            self.assertEqual(m.signal_too_old({"tf": tf, "bars_ago": bars}), old, (tf, bars))

    def test_backtest_does_not_use_the_cutoff(self):
        import inspect
        for fn in (m.backtest_pair, m.run_backtest, m.signals_for_tf, m.detect_wave_divergence):
            self.assertNotIn("signal_too_old", inspect.getsource(fn), fn.__name__)
        self.assertIn("signal_too_old", inspect.getsource(m.scan_route))


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

    def test_level_labels_are_nudged_apart_in_their_own_order(self):
        self.assertEqual(m.spread_labels([100.0, 100.2, 109.0], 2.0), [100.0, 102.0, 109.0])
        self.assertEqual(m.spread_labels([109.0, 100.2, 100.0], 2.0), [109.0, 102.0, 100.0])
        ys = sorted(m.spread_labels([1.0, 1.0, 1.0, 1.0, 1.0], 0.5))
        self.assertTrue(all(b - a >= 0.5 for a, b in zip(ys, ys[1:])), ys)
        self.assertEqual(m.spread_labels([5.0, 1.0, 9.0], 2.0), [5.0, 1.0, 9.0])   # nothing collides

    def boxes(self, hit, narrow=False):
        """Per panel: the A / B value labels, the right-hand level labels, and the
        A-to-B line, all in pixels as the PNG is drawn."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = m.chart_figure(*hit, narrow=narrow)
        FigureCanvasAgg(fig)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        out = []
        for ax in fig.axes:
            ab = [t.get_window_extent(r) for t in ax.texts if t.get_text().split(" ")[0] in ("A", "B")]
            lv = [t.get_window_extent(r) for t in ax.texts if t.get_text().split(" ")[0] not in ("A", "B")]
            seg = next(l for l in ax.lines if l.get_marker() == "o" and len(l.get_xdata()) == 2)
            out.append((ab, lv, ax.transData.transform(seg.get_xydata()), ax.get_window_extent(r)))
        return fig, out

    def variants(self):
        """The fixture as found, its bearish mirror, and each with B squeezed up against A,
        with B on the last bar, and at a sub-cent price (long labels)."""
        s = m.signals_for_tf(wave_df(LONG, side="bearish"), "1h", m.STRICTNESS["balanced"],
                             99.5, "ENS/USDT:USDT", 9e7)[0]
        for df, d, sig in (self.hit, (s.pop("_df"), s.pop("_d"), s)):
            yield df, d, sig
            yield df, dict(d, b_p=d["a_p"] + 2, b_m=d["a_m"] + 2), sig
            yield df, dict(d, b_p=len(df) - 1, b_m=len(df) - 1), sig
            yield df, dict(d, a_price=d["a_price"] * 1e-7, b_price=d["b_price"] * 1e-7), sig

    def test_a_and_b_labels_touch_neither_each_other_nor_their_line(self):
        for hit, narrow in [(h, n) for h in self.variants() for n in (False, True)]:
            for ab, _, ((x0, y0), (x1, y1)), panel in self.boxes(hit, narrow)[1]:
                self.assertEqual(len(ab), 2)
                for b in ab:      # inside the panel: not over the axis ticks, not over the level labels
                    self.assertTrue(panel.x0 <= b.x0 and b.x1 <= panel.x1 and panel.y0 <= b.y0 and b.y1 <= panel.y1,
                                    (hit[1]["side"], b, panel))
                self.assertFalse(ab[0].overlaps(ab[1]), (hit[1]["side"], ab))
                for k in range(101):
                    pt = (x0 + (x1 - x0) * k / 100, y0 + (y1 - y0) * k / 100)
                    self.assertFalse(any(b.contains(*pt) for b in ab), (hit[1]["side"], pt))

    def test_level_labels_do_not_overlap_on_the_drawn_chart(self):
        for hit, narrow in [(h, n) for h in self.variants() for n in (False, True)]:
            lv = self.boxes(hit, narrow)[1][0][1]
            self.assertEqual(len(lv), 5)
            if hit[1]["a_price"] < 1:      # the sub-cent variant only stretches the A / B labels
                continue
            for i, a in enumerate(lv):
                for b in lv[i + 1:]:
                    self.assertFalse(a.overlaps(b), (a, b))

    def test_chart_text_reads_at_12px_in_the_opened_row(self):
        # the chart column of the opened row is 462px wide on a 1280px window
        from matplotlib.text import Text
        # ... and the phone render is drawn 360px wide on a 420px screen
        for narrow, shown in ((False, 462), (True, 360)):
            fig = self.boxes(self.hit, narrow)[0]
            sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
            self.assertGreaterEqual(min(sizes) / 72 * shown / fig.get_figwidth(), 12, (narrow, sorted(set(sizes))))

    def test_phone_chart_keeps_the_a_and_b_values_in_the_panel_titles(self):
        fig = m.chart_figure(*self.hit, narrow=True)
        titles = [ax.get_title(loc="left") for ax in fig.axes]
        d = self.hit[1]
        self.assertIn("A %s" % m._plain(m.px(d["a_price"])), titles[0])
        self.assertIn("B %s" % m._plain(m.px(d["b_price"])), titles[0])
        self.assertIn("A %.3g" % d["a_macd"], titles[1])
        self.assertIn("B %.3g" % d["b_macd"], titles[1])

    def test_route_serves_a_narrower_png_for_phones(self):
        import struct
        q = {"pair": "ENS/USDT:USDT", "side": "bullish", "tf": "1h"}
        wide = self.c.get("/chart", query_string=q).data
        narrow = self.c.get("/chart", query_string=dict(q, w="narrow")).data
        w = [struct.unpack(">I", png[16:20])[0] for png in (wide, narrow)]      # IHDR width
        self.assertLess(w[1], w[0], w)

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
        # the panel sits beside <details>, not inside it, so opening it cannot move the summary
        self.assertIn("""ontoggle="document.getElementById('scanSetBody').hidden = !this.open"><summary>Scan settings</summary></details>""", scan)
        panel = re.search(r'<div class="setbody" id="scanSetBody" hidden>(.*?)\n    </div>\n', scan, re.S).group(1)
        for cid in ("exchange", "minVol", "uniBtn", "uniOut", "strG", "strHint", "maxPairs", "spdG",
                    "symOverride"):
            self.assertIn('id="%s"' % cid, panel, cid)
        for cid in ("runBtn", "pgSec", "pbFill", "pbLbl", "lc", "sl", "scan-audit"):
            self.assertIn('id="%s"' % cid, scan, cid)
        self.assertNotIn("<aside", self.html)

    def test_the_look(self):
        for token in ("--bg:#0C111B", "--sf:#121926", "--sf2:#182132", "--line:#222D40", "--tx:#DCE3EE",
                      "--tx2:#9AA8BD", "--tx3:#6E7D94", "--acc:#8FB4E8",
                      "--yes:#8FBF9F", "--yes-bg:#1B2A24", "--wait:#D8BE8A", "--wait-bg:#2B2619",
                      "--no:#C9A0A5", "--no-bg:#2A2023", "--short:#D99AA0", "--short-bg:#2E1F23",
                      "--blue:#9DB7DA", "--blue-bg:#1C2636", "--violet:#B3A6D6", "--violet-bg:#252238",
                      "--teal:#8CC2C0", "--teal-bg:#18292B",
                      "font-size:15px", ":focus-visible", "prefers-reduced-motion"):
            self.assertIn(token, self.html, token)
        # nothing below 12px (the Method drawings are SVG user units, scaled up on screen)
        css = "\n".join(l for l in self.html.splitlines() if not l.startswith(".lp-"))
        small = [v for v in re.findall(r"font-size:\s*([\d.]+)px", css) if float(v) < 12]
        self.assertEqual(small, [])
        # the old palette is gone
        for gone in ("var(--bull", "var(--bear", "var(--fresh", "var(--bd", "var(--sf3"):
            self.assertNotIn(gone, self.html, gone)

    def test_matte_pills_chips_and_tag_families(self):
        for want in (".side-long{", ".side-short{", ".gr-A{", ".gr-B{", ".gr-C{", ".oi-up{", ".oi-dn{",
                     ".tag.t-pattern{", ".tag.t-flow{", ".tag.t-mover{", ".tag.t-caution{", ".tag.t-venue{",
                     ".mkt-bear{", ".mkt-bull{", ".mkt-chop{"):
            self.assertIn(want, self.html, want)
        self.assertNotIn("Kevin", self.html)

    def test_text_on_the_tints_keeps_4_5_to_1(self):
        tok = dict(re.findall(r"--([a-z0-9-]+):(#[0-9A-Fa-f]{6})", self.html))

        def lum(h):
            c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            r, g, b = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        for fg, bg in [(k, k + "-bg") for k in ("yes", "wait", "no", "short", "blue", "violet", "teal")] + \
                      [("tx", "short-bg"), ("tx", "yes-bg"), ("tx", "wait-bg"), ("tx2", "sf2"), ("acc-ink", "acc")]:
            hi, lo = sorted((lum(tok[fg]), lum(tok[bg])), reverse=True)
            self.assertGreaterEqual((hi + 0.05) / (lo + 0.05), 4.5, (fg, bg))

    def test_rows_stack_on_a_phone(self):
        for want in ("@media (max-width:640px)", 'data-l="Entry"', 'data-l="Stop"', 'data-l="7d OI"',
                     "content:attr(data-l)"):
            self.assertIn(want, self.html, want)

    def test_old_signal_cutoff_is_explained(self):
        for want in ("older than 3 days hidden", "more than 72 hours ago", "m.old_hidden"):
            self.assertIn(want, self.html, want)
        tip = re.search(r"'PARTIAL DATA':\s*'([^']*)'", self.html).group(1)
        self.assertIn("open interest", tip)
        self.assertIn("first bar of wave B", tip)
        self.assertIn("&w=narrow", self.html)

    def test_pivot_era_ui_is_gone(self):
        for gone in ("macd_cross", "volCell", "FROM CURRENT PRICE", "MACD pivot must align"):
            self.assertNotIn(gone, self.html, gone)


# The page script run in node against a stub DOM: enough to call its pure helpers
# and read back what the brief writes.
STUB = """
var P = new Proxy(function(){}, {get: function(){ return P; }, apply: function(){ return P; }, set: function(){ return true; }});
var els = {};
var document = {getElementById: function(id){ return els[id] || (els[id] = {style: {}, dataset: {}, classList: P,
                  addEventListener: function(){}, setAttribute: function(){}, querySelectorAll: function(){ return []; }}); },
                querySelectorAll: function(){ return []; }, querySelector: function(){ return P; }, addEventListener: function(){}};
var location = {hash: ''}, fetch = P;
globalThis.setInterval = function(){};
"""
ROWS = """
function row(base, v, reason, extra){
  var s = Object.assign({grade: 'A', score: 74, regime: 'WITH TREND', status: 'active', bars_ago: 1, fresh: true,
    forming: false, trigger_first: false, entry_ref_price: 10, stop_price: 9.5, risk_pct: 5, tags: ['TRIPLE'],
    targets: {'1r': 10.5, '2r': 11, '3r': 11.5, structure: 10.8}}, extra || {});
  return {pair: base + '/USDT:USDT', base: base, side: 'bullish', tfs: ['4h'], primary_tf: '4h', signals: {'4h': s},
          score: s.score, current_price: 10.1, entry: {verdict: v, reason: reason, level: v === 'skip' ? null : 9.5}};
}
var YES = row('AAA', 'enter', 'Yes (confirmed). The signal bar has closed, the grade is A and the stop is untouched. Risk 5% to the stop at');
var BEST = row('BBB', 'enter', 'Yes (early). The lower timeframe triggered while the signal bar is still forming. Risk 5% to the stop at', {score: 90, forming: true, trigger_first: true});
var WAIT = row('CCC', 'wait', 'Not yet. Price has already moved 3% away from the signal. Wait for it to pull back to', {grade: 'B', score: 60, status: 'extended', pct_chg: 3});
var NO = row('DDD', 'skip', 'No. Weak setup: grade C, score 40 of 100.', {grade: 'C', score: 40});
function brief(rows, meta){ pendR = rows; scanMeta = meta; scanTs = Date.now(); renderBrief();
  return {h: els.briefH.textContent, lbl: els.briefLbl.textContent, state: els.briefState.innerHTML, picks: els.picks.innerHTML}; }
"""


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class PageLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        html = m.app.test_client().get("/").get_data(as_text=True)
        cls.js = STUB + re.search(r"<script>(.*?)</script>", html, re.S).group(1) + ROWS

    def js_value(self, expr):
        path = os.path.join(tempfile.mkdtemp(), "page.js")
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.js + "\nprocess.stdout.write(JSON.stringify(" + expr + "));")
        out = subprocess.run(["node", path], capture_output=True, text=True, encoding="utf-8", timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_tag_family_comes_from_the_tags_page_groups(self):
        got = self.js_value("[tag('TRIPLE'), tag('CAPITULATION'), tag('HIGH OI/VOL'), tag('BIG MOVE'), "
                            "tag('PARTIAL DATA'), tag('MACRO', 'MACRO CPI m/m in 3h'), tag('NOT ON BINANCE'), "
                            "tag('ALSO ON', 'ALSO ON OKX'), tag('WITH TREND')]")
        fams = [(re.search(r'class="tag t-(\w+)"', h) or [None, None])[1] for h in got]
        self.assertEqual(fams, ["pattern", "flow", "flow", "mover", "caution", "caution", "caution", "venue",
                                None])
        self.assertIn(">MACRO CPI m/m in 3h<", got[5])

    def test_answer_cell_shows_the_first_sentence_only(self):
        self.assertEqual(self.js_value("[shortWhy(YES), shortWhy(BEST), shortWhy(WAIT), shortWhy(NO)]"),
                         ["The signal bar has closed, the grade is A and the stop is untouched",
                          "The lower timeframe triggered while the signal bar is still forming",
                          "Price has already moved 3% away from the signal",
                          "Weak setup: grade C, score 40 of 100"])
        self.assertTrue(self.js_value("reason(WAIT)").endswith("pull back to 9.5"))

    def test_the_pill_says_which_yes(self):
        got = self.js_value("[ansPill(YES), ansPill(BEST), ansPill(WAIT), ansPill(NO), evCell(BEST)]")
        self.assertEqual([re.sub(r"<[^>]+>", "", h) for h in got[:4]], ["Yes", "Yes (early)", "Not yet", "No"])
        self.assertEqual(re.findall(r'class="ans (\w+)"', "".join(got)), ["enter", "enter", "wait", "skip", "enter"])
        self.assertIn(">Yes (early)</span>", got[4])

    def test_brief_with_yes_rows(self):
        b = self.js_value("brief([NO, WAIT, YES, BEST], {btc: {state: 'BEAR', rsi: 32.4, close: 61959.3, ema200: 73736.56},"
                          " blocker: {ts_ms: Date.now() + 3 * 3600000 + 10000, title: 'CPI m/m'}, macro: null})")
        self.assertIn("2 setups pass the checklist; the rest can wait.", b["h"])
        self.assertEqual(b["lbl"], "Worth a look")
        self.assertEqual(re.findall(r'class="coin">(\w+)<', b["picks"]), ["BBB", "AAA"])   # highest score first
        self.assertEqual(re.findall(r'class="ans enter">([^<]+)<', b["picks"]), ["Yes (early)", "Yes"])
        self.assertIn('class="mkt mkt-bear"', b["state"])
        self.assertIn("61959.30</span>, below its 200 EMA at <span class=\"num\">73736.56<", b["state"])
        self.assertIn("<b>CPI m/m in 3h 0m.</b> All entries wait until an hour after it prints.", b["state"])

    def test_brief_with_only_not_yet_rows_and_a_far_print(self):
        b = self.js_value("brief([NO, WAIT], {btc: null,"
                          " blocker: {ts_ms: Date.now() + 8 * 3600000 + 10000, title: 'FOMC Statement'},"
                          " macro: {ts_ms: Date.now() + 2 * 3600000 + 10000, title: 'BOJ Policy Rate'}})")
        self.assertIn("Nothing passes yet. 1 setup is close.", b["h"])
        self.assertEqual(b["lbl"], "Closest to qualifying")
        self.assertIn("BTC data is unavailable.", b["state"])
        self.assertIn("4H setups wait until it is out. 1H setups are clear for now.", b["state"])
        self.assertIn("Also due: BOJ Policy Rate in 2h 0m.", b["state"])

    def test_brief_of_a_scan_saved_before_the_brief_says_nothing_about_market_or_macro(self):
        b = self.js_value("brief([NO], {})")
        self.assertIn("No trade is a fine answer.", b["h"])
        self.assertEqual((b["state"], re.findall(r'class="coin">(\w+)<', b["picks"])), ("", ["DDD"]))

    def test_audit_line_counts_hidden_old_signals(self):
        got = self.js_value("[auditLine({dispatched: 9, scanned: 7, old_hidden: 5}), auditLine({dispatched: 9, scanned: 7}),"
                            " auditLine({dispatched: 9, scanned: 7, old_hidden: 1})]")
        self.assertIn("5 signals older than 3 days hidden", got[0])
        self.assertNotIn("hidden", got[1])       # a scan saved before the cutoff, or nothing hidden
        self.assertIn("1 signal older than 3 days hidden", got[2])

    def test_checks_follow_the_row(self):
        got = self.js_value("(scanMeta = {blocker: {ts_ms: 3 * 3600000, title: 'CPI m/m'}}, scanTs = 1,"
                            " [checks(YES), checks(WAIT), checks(NO)])")
        self.assertEqual([c[0] for c in got[0]], ["", "", "", "", "w"])       # the print is inside the 12h window
        self.assertEqual([c[0] for c in got[1]], ["", "", "", "w", "w"])
        self.assertEqual(got[2][1], ["x", "Grade C (40) is too weak"])


if __name__ == "__main__":
    unittest.main()
