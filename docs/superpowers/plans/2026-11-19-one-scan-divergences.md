# One Scan, One Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the screener into one scan and one table of 4H and 1H MACD divergences, each row carrying an entry verdict and Movers / OI-flow context badges.

**Architecture:** Everything stays in `macd_divergence_screener_v5.py` (Flask app plus the dashboard as one raw HTML string). Detection gets a shorter right-side pivot window and tighter geometry. `/scan` fetches 4H then 1H per pair, groups the signals into rows by pair and side, and fetches daily candles and OI history only for pairs that have a divergence. The Watchlist, Movers and Routine pages and their routes are deleted; their classification functions are reused for badges.

**Tech Stack:** Python 3.9+, Flask, ccxt, pandas, numpy, `unittest`. Vanilla JS in the embedded page. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-19-one-scan-divergences-design.md`

## Global Constraints

- Tests run with `python -m unittest discover -s tests -v`. There is no pytest.
- No new dependencies and no new source files other than tests.
- Commit messages are plain. Do NOT add `Co-Authored-By` or "Generated with Claude Code" lines.
- Work on branch `feature/one-scan-divergences`. Do not push.
- The HTML is a raw string (`HTML = r"""..."""`). Never introduce `"""` inside it.
- Timeframes are exactly `4h` then `1h`, in that order.
- Pivot windows: 5 bars left, 3 bars right, on both timeframes.
- 4H: `max_dist` 60, `max_age` 30, `fresh` 12. 1H: `max_dist` 60, `max_age` 24, `fresh` 12, 500 bars.
- Balanced `min_macd_frac` is 0.15. Loose strictness and the 1D timeframe do not exist.
- A macro "market mover" within 24 hours makes a verdict `wait`.
- Verdict values are exactly `enter`, `wait`, `skip`.
- Between Task 5 and Task 6 the browser page is broken on purpose (server changed, page not yet). Tests must still pass after every task.

## File Structure

| File | Change |
|---|---|
| `macd_divergence_screener_v5.py` | All production changes |
| `tests/test_divergence.py` | Create: pivots, detection geometry, `signals_for_tf` |
| `tests/test_rows.py` | Create: `entry_verdict`, `next_mover`, `flow_context`, `build_rows` |
| `tests/test_app.py` | Create: routes, saved-state loading, page smoke test |
| `tests/test_movers.py` | Modify: drop the thin-band test |
| `README.md` | Modify: tabs, API, parameters, reading a row |

---

### Task 1: Pivot window and detection geometry

**Files:**
- Modify: `macd_divergence_screener_v5.py` (`TF_CFG` ~L103, `STRICTNESS` ~L111, `find_pivots` ~L329, `swing_pivots` ~L352, `detect_divergences` ~L406-411)
- Test: `tests/test_divergence.py` (create)

**Interfaces:**
- Produces: `find_pivots(vals, lb, rb=None) -> (highs, lows)`; `swing_pivots(high, low, lb, rb=None)`; `TF_CFG` with keys `"4h"`, `"1h"`, each having `pivot_rb`; `STRICTNESS` with keys `"balanced"`, `"strict"`.
- Produces for later tests: `tri()` and `make_df()` helpers in `tests/test_divergence.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_divergence.py`:

```python
"""Tests for pivot detection and divergence geometry.

Run with:  python -m unittest tests.test_divergence -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402


def tri(n, c, depth, w=6):
    """A triangle of height `depth` centred on bar c, zero elsewhere."""
    t = np.arange(n)
    return depth * np.clip(1 - np.abs(t - c) / w, 0, None)


def make_df(a, b, side="bullish", n=120, freq="1h"):
    """Hand-built candles AND indicators with one clean regular divergence.

    Bullish: price low at b is lower than at a, MACD low at b is higher.
    Bearish is the mirror. Indicators are supplied directly so the test
    controls the geometry instead of fighting an EMA.
    """
    sgn = -1 if side == "bullish" else 1
    ext = 100 + sgn * (tri(n, a, 5) + tri(n, b, 8))
    macd = sgn * (tri(n, a, 2.0) + tri(n, b, 1.0))
    if side == "bullish":
        low, high = ext, ext + 1
    else:
        high, low = ext, ext - 1
    idx = pd.date_range("2026-01-01", periods=n, freq=freq)
    return pd.DataFrame({"open": low + .5, "high": high, "low": low, "close": low + .5,
                         "volume": 1.0, "macd": macd, "signal": macd - sgn * 0.1,
                         "atr": 1.0}, index=idx)


class Pivots(unittest.TestCase):
    VALS = np.array([10.0] * 10 + [5.0] + [10.0] * 3)   # low at bar 10, 3 bars after it

    def test_pivot_is_confirmed_after_rb_bars(self):
        self.assertEqual(m.find_pivots(self.VALS, 5, 3)[1], [10])

    def test_default_right_window_equals_left(self):
        self.assertEqual(m.find_pivots(self.VALS, 5)[1], [])

    def test_pivot_does_not_move_when_bars_are_appended(self):
        longer = np.append(self.VALS, [10.0] * 5)
        self.assertEqual(m.find_pivots(longer, 5, 3)[1], [10])


class Config(unittest.TestCase):
    def test_timeframes_are_4h_then_1h(self):
        self.assertEqual(list(m.TF_CFG), ["4h", "1h"])

    def test_loose_is_gone(self):
        self.assertEqual(sorted(m.STRICTNESS), ["balanced", "strict"])


class Detection(unittest.TestCase):
    RULES = m.STRICTNESS["balanced"]

    def detect(self, a, b, side, tf):
        return m.detect_divergences(make_df(a, b, side), m.TF_CFG[tf], self.RULES)

    def test_clear_divergence_is_found_on_both_timeframes(self):
        for tf in ("4h", "1h"):
            for side in ("bullish", "bearish"):
                out = self.detect(60, 100, side, tf)
                self.assertEqual([(s["side"], s["pivot_dist"]) for s in out],
                                 [(side, 40)], (tf, side))

    def test_listed_three_bars_after_the_second_pivot(self):
        out = self.detect(70, 116, "bullish", "4h")     # 120 bars: 3 bars follow bar 116
        self.assertEqual([(s["bars_ago"], s["fresh"]) for s in out], [(3, True)])

    def test_second_pivot_older_than_max_age_is_dropped(self):
        for tf in ("4h", "1h"):
            self.assertEqual(self.detect(20, 60, "bullish", tf), [])   # 59 bars old

    def test_pivots_further_apart_than_max_dist_are_dropped(self):
        for tf in ("4h", "1h"):
            self.assertEqual(self.detect(20, 100, "bearish", tf), [])  # 80 bars apart


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_divergence -v`
Expected: FAIL. `find_pivots() takes 2 positional arguments but 3 were given`, `KeyError: '1h'`, and the `Config` assertions.

- [ ] **Step 3: Implement**

Replace `TF_CFG` and `STRICTNESS`:

```python
# Per-timeframe geometry. A pivot must be the extreme of pivot_lb bars to its
# left but only pivot_rb to its right: the right side is pure waiting time
# (5 bars on 4H is 20 hours), and a divergence listed a day late is not usable.
TF_CFG = {
    "4h": {"label": "4H", "minutes": 240, "bars": 720, "pivot_lb": 5, "pivot_rb": 3,
           "min_dist": 5, "max_dist": 60, "max_age": 30, "fresh": 12},
    "1h": {"label": "1H", "minutes": 60,  "bars": 500, "pivot_lb": 5, "pivot_rb": 3,
           "min_dist": 5, "max_dist": 60, "max_age": 24, "fresh": 12},
}

# Strictness presets — see the module docstring for what each rule does.
STRICTNESS = {
    "balanced": {"require_macd_pivot": True,  "require_clean": True,
                 "min_price_pct": 0.25, "atr_frac": 0.20, "min_macd_frac": 0.15,
                 "zero_line": False},
    "strict":   {"require_macd_pivot": True,  "require_clean": True,
                 "min_price_pct": 0.50, "atr_frac": 0.55, "min_macd_frac": 0.22,
                 "zero_line": True},
}
```

Replace the body of `find_pivots` (keep the first two docstring paragraphs, replace the third):

```python
def find_pivots(vals, lb, rb=None):
    """Indices that are the extreme of a window of lb bars before and rb after.

    Ties resolve to the EARLIEST bar (np.argmax/argmin return the first
    occurrence), so a flat double-bottom produces exactly one pivot instead of
    zero. v3 required a strictly unique extreme and silently dropped every
    flat-bottomed swing — common on low-tick-size alts.

    A pivot needs rb CLOSED bars after it, so the newest possible pivot is rb
    bars old. That is what makes this non-repainting: a pivot never appears,
    moves, or disappears on a later scan. rb defaults to lb.
    """
    rb = lb if rb is None else rb
    n = len(vals)
    highs, lows = [], []
    for i in range(lb, n - rb):
        w = vals[i - lb:i + rb + 1]
        if i - lb + int(np.argmax(w)) == i:
            highs.append(i)
        if i - lb + int(np.argmin(w)) == i:
            lows.append(i)
    return highs, lows
```

In `swing_pivots`, change the signature and the two calls (docstring unchanged):

```python
def swing_pivots(high, low, lb, rb=None):
    ...
    ph, _ = find_pivots(high, lb, rb)
    _, pl = find_pivots(low, lb, rb)
    return ph, pl
```

In `detect_divergences`, change the pivot lines:

```python
    lb, rb = cfg["pivot_lb"], cfg["pivot_rb"]
    mn, mx, age = cfg["min_dist"], cfg["max_dist"], cfg["max_age"]

    price_h, price_l = swing_pivots(high, low, lb, rb)
    macd_h,  macd_l  = find_pivots(macd, lb, rb)
```

- [ ] **Step 4: Run all tests**

Run: `python -m unittest discover -s tests -v`
Expected: all PASS (new tests plus the existing calendar and mover tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_divergence.py macd_divergence_screener_v5.py
git commit -m "Confirm pivots after 3 bars and tighten divergence geometry for 4H and 1H"
```

---

### Task 2: Entry verdict

**Files:**
- Modify: `macd_divergence_screener_v5.py` (add after `setup_score`, before `# ── Exchange helpers`)
- Test: `tests/test_rows.py` (create)

**Interfaces:**
- Consumes: signal dict keys `status`, `fresh`, `bars_ago`, `r_reached`, `pct_chg`, `signal_price`, `macd_cross`, `stop_price`, `risk_pct` (all produced today by `_mk`, `evaluate_status`, `build_targets`); calendar events `{"ts": "...Z", "mover": bool, "title": str}`; existing `_cal_parse_ts("iso", ts)`.
- Produces: `EVENT_WAIT_H = 24`; `entry_verdict(sig, hours_to_event=None, event_title="") -> {"verdict": "enter"|"wait"|"skip", "reason": str, "level": float|None}`; `next_mover(events, now) -> (hours: float|None, title: str)`.
- `level` is a price the page prints after the reason with its own formatter (Python would print sub-cent prices in exponent form). Reasons that carry a level end with the word that introduces it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rows.py`:

```python
"""Tests for the entry verdict, context and row grouping.

Run with:  python -m unittest tests.test_rows -v
"""
import datetime as dt
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402

SIG = {"status": "active", "fresh": True, "bars_ago": 2, "r_reached": 0.0,
       "pct_chg": 0.4, "signal_price": 100.0, "macd_cross": True,
       "stop_price": 97.0, "risk_pct": 3.0}


def sig(**kw):
    return dict(SIG, **kw)


class EntryVerdict(unittest.TestCase):
    def verdict(self, s, *a):
        return m.entry_verdict(s, *a)["verdict"]

    def test_enter_when_every_check_passes(self):
        v = m.entry_verdict(sig())
        self.assertEqual((v["verdict"], v["level"]), ("enter", 97.0))
        self.assertIn("risk 3.0%", v["reason"])

    def test_invalidated_is_skip(self):
        v = m.entry_verdict(sig(status="invalidated"))
        self.assertEqual((v["verdict"], v["reason"]), ("skip", "Stop was hit"))

    def test_ran_is_skip(self):
        v = m.entry_verdict(sig(status="ran", r_reached=2.5))
        self.assertEqual((v["verdict"], v["reason"]), ("skip", "Already ran 2.5R"))

    def test_not_fresh_is_skip(self):
        v = m.entry_verdict(sig(fresh=False, bars_ago=20))
        self.assertEqual((v["verdict"], v["reason"]), ("skip", "Too old (20 bars)"))

    def test_extended_waits_for_a_pullback_to_the_signal_price(self):
        v = m.entry_verdict(sig(status="extended", pct_chg=3.1))
        self.assertEqual((v["verdict"], v["level"]), ("wait", 100.0))
        self.assertIn("3.1%", v["reason"])

    def test_no_macd_cross_is_wait(self):
        self.assertEqual(self.verdict(sig(macd_cross=False)), "wait")

    def test_event_inside_24h_is_wait_and_names_the_event(self):
        v = m.entry_verdict(sig(), 14.2, "CPI m/m")
        self.assertEqual((v["verdict"], v["reason"]), ("wait", "CPI m/m in 14h"))

    def test_event_beyond_24h_or_unknown_does_not_block(self):
        self.assertEqual(self.verdict(sig(), 30.0, "CPI m/m"), "enter")
        self.assertEqual(self.verdict(sig(), None), "enter")

    def test_skip_outranks_wait(self):
        self.assertEqual(self.verdict(sig(status="invalidated"), 2.0, "FOMC"), "skip")


class NextMover(unittest.TestCase):
    NOW = dt.datetime(2026, 11, 19, 12, 0, tzinfo=dt.timezone.utc)
    EVENTS = [
        {"ts": "2026-11-19T10:00:00Z", "mover": True,  "title": "Past print"},
        {"ts": "2026-11-19T15:00:00Z", "mover": False, "title": "Not a mover"},
        {"ts": "2026-11-20T13:30:00Z", "mover": True,  "title": "CPI m/m"},
        {"ts": "2026-11-21T19:00:00Z", "mover": True,  "title": "FOMC Statement"},
    ]

    def test_nearest_future_mover(self):
        self.assertEqual(m.next_mover(self.EVENTS, self.NOW), (25.5, "CPI m/m"))

    def test_none_when_nothing_is_scheduled(self):
        self.assertEqual(m.next_mover([], self.NOW), (None, ""))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m unittest tests.test_rows -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'entry_verdict'` (and `next_mover`).

- [ ] **Step 3: Implement**

Add after `setup_score`:

```python
# ── Entry verdict ──────────────────────────────────────────────────────────────
EVENT_WAIT_H = 24     # a curated macro print inside this many hours means wait


def entry_verdict(sig, hours_to_event=None, event_title=""):
    """Is this setup enterable as of the scan? A CHECKLIST, not a prediction.

    First matching rule wins, dead setups before waiting ones:
      skip   stop hit · already ran 1R+ · second pivot outside the fresh window
      wait   price ran >MOVED_PCT from the signal · MACD has not crossed since
             the pivot · a market-mover print is inside EVENT_WAIT_H
      enter  none of the above

    `level` is a price the dashboard prints after the reason, so it is
    formatted in one place (sub-cent prices need plain digits).
    """
    def out(verdict, reason, level=None):
        return {"verdict": verdict, "reason": reason, "level": level}

    st = sig["status"]
    if st == "invalidated":
        return out("skip", "Stop was hit")
    if st == "ran":
        return out("skip", f"Already ran {sig['r_reached']:g}R")
    if not sig["fresh"]:
        return out("skip", f"Too old ({sig['bars_ago']} bars)")
    if st == "extended":
        return out("wait", f"Moved {sig['pct_chg']:g}% from signal, wait for a pullback to",
                   sig["signal_price"])
    if not sig["macd_cross"]:
        return out("wait", "Momentum has not turned, wait for the MACD cross")
    if hours_to_event is not None and hours_to_event <= EVENT_WAIT_H:
        return out("wait", f"{event_title} in {hours_to_event:.0f}h")
    return out("enter", f"Near signal, MACD crossed, risk {sig['risk_pct']}%, stop",
               sig["stop_price"])


def next_mover(events, now):
    """(hours until the next curated market mover, its title), or (None, "")."""
    best = (None, "")
    for e in events:
        if not e.get("mover"):
            continue
        ts = _cal_parse_ts("iso", e.get("ts"))
        if ts is None or ts <= now:
            continue
        h = (ts - now).total_seconds() / 3600.0
        if best[0] is None or h < best[0]:
            best = (h, e.get("title", ""))
    return best
```

- [ ] **Step 4: Run the tests**

Run: `python -m unittest tests.test_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rows.py macd_divergence_screener_v5.py
git commit -m "Add the entry verdict and next-market-mover lookup"
```

---

### Task 3: Context for a pair (Movers and OI flow)

**Files:**
- Modify: `macd_divergence_screener_v5.py` (add after `classify_flow`, ~L899; add `pair_context` after `drop_forming_candle`, ~L1521, because it needs `fetch_ohlcv`, `_to_df` and `drop_forming_candle`)
- Test: `tests/test_rows.py` (append)

**Interfaces:**
- Consumes: existing `classify_flow`, `oi_point_usd`, `fetch_oi_history`, `analyse_mover`, `classify_mover`, `fetch_ohlcv`, `_to_df`, `drop_forming_candle`, constants `OI_*`, `MOVER_LOOKBACK_D`.
- Produces: `flow_context(hist, df, live_price, fund_day, wk_vol) -> dict | None` with keys `quadrant`, `flags`, `oi_chg_pct`, `price_chg_pct`, `funding_day_pct`; `pair_context(ex, limiter, sym, live_price, fund_day, wk_vol) -> (mover: dict|None, flow: dict|None, error: str|None)` where `mover` is `analyse_mover`'s dict plus `"flags"`.
- `flow_context` is the per-pair body of `watchlist_route`, moved verbatim into a pure function. `watchlist_route` itself is deleted in Task 5; do not refactor it to call this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rows.py` before the `if __name__` block:

```python
def oi_hist(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="D")
    return [{"timestamp": int(t.timestamp() * 1000), "openInterestValue": v}
            for t, v in zip(idx, values)]


def daily(closes):
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"close": closes}, index=idx)


class FlowContext(unittest.TestCase):
    RISING = [100, 104, 108, 112, 116, 120, 125, 130]      # +30% over 7 days
    UP = [10, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.5]    # +15%

    def test_new_longs_with_flat_funding_is_spot_led(self):
        f = m.flow_context(oi_hist(self.RISING), daily(self.UP), 11.5, 0.0, 7e9)
        self.assertEqual((f["quadrant"], f["flags"], f["oi_chg_pct"]),
                         ("NEW LONGS", ["SPOT-LED"], 30.0))

    def test_oi_above_daily_volume_multiple_is_flagged(self):
        f = m.flow_context(oi_hist(self.RISING), daily(self.UP), 11.5, None, 70.0)
        self.assertEqual(f["flags"], ["HIGH OI/VOL"])      # OI 130 vs $10/day volume

    def test_quiet_oi_gives_no_context(self):
        flat = [100, 100, 101, 100, 101, 100, 101, 102]
        self.assertIsNone(m.flow_context(oi_hist(flat), daily(self.UP), 11.5, 0.0, 7e9))

    def test_short_history_gives_no_context(self):
        self.assertIsNone(m.flow_context(oi_hist([100, 130]), daily(self.UP), 11.5, 0.0, 7e9))
        self.assertIsNone(m.flow_context(None, daily(self.UP), 11.5, 0.0, 7e9))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m unittest tests.test_rows -v`
Expected: `FlowContext` tests FAIL with `AttributeError: ... 'flow_context'`.

- [ ] **Step 3: Implement**

Add after `classify_flow`:

```python
def flow_context(hist, df, live_price, fund_day, wk_vol):
    """OI-flow read for one pair from its daily OI history and daily candles.

    None means "nothing to say": too little history (a fresh listing's OI
    always ramps from zero), no baseline point near the lookback, or OI that
    moved less than OI_MIN_CHANGE_PCT.
    """
    hist = sorted((p for p in (hist or []) if p.get("timestamp")),
                  key=lambda p: p["timestamp"])
    if len(hist) < OI_MIN_HISTORY:
        return None

    # Baseline = the OI point closest to (now − lookback). Daily points can be
    # sparse on some venues, so "closest" with a ±36h tolerance beats indexing
    # back a fixed count.
    now_ms = hist[-1]["timestamp"]
    target = now_ms - OI_LOOKBACK_D * 86_400_000
    past = min(hist[:-1], key=lambda p: abs(p["timestamp"] - target))
    if abs(past["timestamp"] - target) > 36 * 3_600_000:
        return None

    # price for amount→USD conversion at the baseline: that day's close
    day_close = {int(ts.timestamp() * 1000) // 86_400_000: float(c)
                 for ts, c in df["close"].items()}
    past_px = day_close.get(past["timestamp"] // 86_400_000, live_price)
    oi_now  = oi_point_usd(hist[-1], live_price)
    oi_then = oi_point_usd(past, past_px)
    if not oi_now or not oi_then:
        return None

    oi_chg = (oi_now - oi_then) / oi_then * 100.0
    if abs(oi_chg) < OI_MIN_CHANGE_PCT:
        return None
    span = min(OI_LOOKBACK_D + 1, len(df))
    price_chg = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-span]) - 1) * 100.0

    quad, flags = classify_flow(price_chg, oi_chg, fund_day)
    if wk_vol and oi_now / (wk_vol / 7.0) >= OI_VOL_RATIO_HIGH:
        flags.append("HIGH OI/VOL")
    return {"quadrant": quad, "flags": flags,
            "oi_chg_pct": round(oi_chg, 1), "price_chg_pct": round(price_chg, 1),
            "funding_day_pct": round(fund_day, 4) if fund_day is not None else None}
```

Add after `drop_forming_candle`:

```python
def pair_context(ex, limiter, sym, live_price, fund_day, wk_vol):
    """Movers + OI-flow context for a pair that HAS a divergence.

    Returns (mover, flow, error). Two calls, spent only on pairs worth a row.
    A failure here must never cost the row itself, so it comes back as text.
    """
    try:
        df = drop_forming_candle(
            _to_df(fetch_ohlcv(ex, limiter, sym, "1d", limit=MOVER_LOOKBACK_D + 12)), 1440)
        if df is None or df.empty:
            return None, None, "no daily candles"
        mover = analyse_mover(df, live_price)
        if mover is not None:
            mover["flags"] = classify_mover(mover)[1]
        hist = fetch_oi_history(ex, limiter, sym, OI_LOOKBACK_D + 4)
        return mover, flow_context(hist, df, live_price, fund_day, wk_vol), None
    except Exception as e:
        return None, None, str(e)
```

- [ ] **Step 4: Run all tests**

Run: `python -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rows.py macd_divergence_screener_v5.py
git commit -m "Add per-pair Movers and OI-flow context for divergence rows"
```

---

### Task 4: Group signals into rows

**Files:**
- Modify: `macd_divergence_screener_v5.py` (add after `dedupe`, ~L1547)
- Test: `tests/test_rows.py` (append)

**Interfaces:**
- Consumes: `entry_verdict` (Task 2), `TF_CFG` key order (Task 1).
- Produces: `build_rows(sigs, mover=None, flow=None, context_error=None, hours_to_event=None, event_title="") -> list[dict]`. `sigs` are the finished signals of ONE pair, each with `tf`, `side`, `pair`, `base`, `current_price`, `weekly_volume`, `score`, `fresh`, `status` plus the verdict keys. Each row has `pair`, `base`, `side`, `current_price`, `weekly_volume`, `tfs` (4h first), `signals` (`{tf: signal}`), `score`, `fresh`, `status`, `entry`, `mover`, `flow`, `context_error`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rows.py` before the `if __name__` block:

```python
def full_sig(tf, side, **kw):
    base = dict(SIG, tf=tf, side=side, pair="ENS/USDT:USDT", base="ENS",
                current_price=100.5, weekly_volume=9e7, score=60)
    return dict(base, **kw)


class BuildRows(unittest.TestCase):
    def test_same_side_on_both_timeframes_is_one_row_4h_first(self):
        rows = m.build_rows([full_sig("1h", "bullish", score=80),
                             full_sig("4h", "bullish", score=70)])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["tfs"], r["score"], sorted(r["signals"])),
                         (["4h", "1h"], 80, ["1h", "4h"]))

    def test_opposite_sides_are_two_rows(self):
        rows = m.build_rows([full_sig("4h", "bullish"), full_sig("1h", "bearish")])
        self.assertEqual(sorted((r["side"], r["tfs"][0]) for r in rows),
                         [("bearish", "1h"), ("bullish", "4h")])

    def test_verdict_and_status_come_from_the_4h_signal(self):
        rows = m.build_rows([full_sig("4h", "bullish", status="invalidated"),
                             full_sig("1h", "bullish")])
        self.assertEqual((rows[0]["status"], rows[0]["entry"]["verdict"], rows[0]["fresh"]),
                         ("invalidated", "skip", True))

    def test_context_and_event_are_carried_onto_the_row(self):
        rows = m.build_rows([full_sig("1h", "bearish")], mover={"flags": ["VOLUME SPIKE"]},
                            flow=None, context_error=None,
                            hours_to_event=3.0, event_title="FOMC Statement")
        r = rows[0]
        self.assertEqual((r["mover"], r["flow"], r["entry"]["reason"]),
                         ({"flags": ["VOLUME SPIKE"]}, None, "FOMC Statement in 3h"))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m unittest tests.test_rows -v`
Expected: `BuildRows` tests FAIL with `AttributeError: ... 'build_rows'`.

- [ ] **Step 3: Implement**

Add after `dedupe`:

```python
def build_rows(sigs, mover=None, flow=None, context_error=None,
               hours_to_event=None, event_title=""):
    """One row per side for ONE pair's signals across timeframes.

    The primary signal — the one whose status, levels and entry verdict the
    table shows — is the highest timeframe present (TF_CFG order).
    """
    order = list(TF_CFG)
    rows = {}
    for s in sorted(sigs, key=lambda s: order.index(s["tf"])):
        r = rows.setdefault(s["side"], {
            "pair": s["pair"], "base": s["base"], "side": s["side"],
            "current_price": s["current_price"], "weekly_volume": s["weekly_volume"],
            "tfs": [], "signals": {},
            "mover": mover, "flow": flow, "context_error": context_error})
        r["tfs"].append(s["tf"])
        r["signals"][s["tf"]] = s
    for r in rows.values():
        primary = r["signals"][r["tfs"][0]]
        r["score"]  = max(s["score"] for s in r["signals"].values())
        r["fresh"]  = any(s["fresh"] for s in r["signals"].values())
        r["status"] = primary["status"]
        r["entry"]  = entry_verdict(primary, hours_to_event, event_title)
    return list(rows.values())
```

- [ ] **Step 4: Run the tests**

Run: `python -m unittest tests.test_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rows.py macd_divergence_screener_v5.py
git commit -m "Group a pair's 4H and 1H signals into one row per side"
```

---

### Task 5: The one scan, saved state, and removed routes

After this task the server is final. The browser page is broken until Task 6.

**Files:**
- Modify: `macd_divergence_screener_v5.py`: `_state` / `_movers_state` (~L299), `MOVERS_STATE_FILE` (~L248), `classify_mover` (~L958), `save_movers` / `load_movers` (~L1004-1030), `save_state` / `load_state` (~L1550), `signals_route`, `watch_signals_route`, `watchlist_route`, `scan_route`, `mover_signals_route`, `movers_route`, the `__main__` block
- Modify: `tests/test_movers.py`, `tests/test_divergence.py`
- Test: `tests/test_app.py` (create)

**Interfaces:**
- Consumes: `build_rows`, `pair_context`, `next_mover`, `TF_CFG`, `fetch_funding_map`, `refresh_calendar`, `_cal_state`.
- Produces: `signals_for_tf(df, tf, rules, live_price, sym, wk_vol) -> list[signal]` where `df` already has indicator columns; `GET /signals -> {"rows": [...], "meta": {...}, "ts": iso|null}`; `/scan` SSE `result` events carry one row each; `meta` has `tfs: ["4h","1h"]` and no `tf`; `classify_mover(m)` with no `thin` parameter.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app.py`:

```python
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
        for path in ("/watchlist", "/watch_signals", "/movers", "/mover_signals"):
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


if __name__ == "__main__":
    unittest.main()
```

Append to `tests/test_divergence.py` before the `if __name__` block:

```python
class SignalsForTf(unittest.TestCase):
    def test_finished_signal_carries_everything_a_row_needs(self):
        df = make_df(70, 116, "bullish")
        out = m.signals_for_tf(df, "1h", m.STRICTNESS["balanced"],
                               float(df["close"].iloc[-1]), "ENS/USDT:USDT", 9e7)
        self.assertEqual(len(out), 1)
        s = out[0]
        self.assertEqual((s["tf"], s["pair"], s["base"], s["side"], s["weekly_volume"]),
                         ("1h", "ENS/USDT:USDT", "ENS", "bullish", 9e7))
        for k in ("status", "targets", "risk_pct", "score", "current_price", "stop_price"):
            self.assertIn(k, s)
        self.assertNotIn("_b", s)

    def test_too_few_bars_gives_nothing(self):
        df = make_df(20, 40, "bullish", n=60)
        self.assertEqual(m.signals_for_tf(df, "4h", m.STRICTNESS["balanced"],
                                          100.0, "ENS/USDT:USDT", 9e7), [])
```

Replace the whole of `tests/test_movers.py` with:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL. `/watchlist` returns 200, `KeyError: 'rows'`, old state is loaded, `signals_for_tf` missing. (`test_movers` passes already; it only lost the thin test.)

- [ ] **Step 3: Add `signals_for_tf`**

Add after `build_rows`:

```python
def signals_for_tf(df, tf, rules, live_price, sym, wk_vol):
    """Finished signals for one pair on one timeframe.

    `df` must already carry the indicator columns. Fewer than 80 bars is not
    enough for MACD to have settled, so that timeframe reports nothing.
    """
    if df is None or len(df) < 80:
        return []
    found = []
    for sig in detect_divergences(df, TF_CFG[tf], rules):
        sig.update(evaluate_status(df, sig, live_price))
        found.append(sig)
    out = []
    for sig in dedupe(found):
        sig.pop("_b", None)
        sig.update(build_targets(sig, live_price))
        sig.update({
            "pair": sym, "base": sym.split("/")[0], "tf": tf,
            "current_price": px(live_price),
            "weekly_volume": round(wk_vol, 2) if wk_vol else None,
            "score": setup_score(sig, wk_vol),
        })
        out.append(sig)
    return out
```

- [ ] **Step 4: Rewrite `scan_route`**

In `scan_route`:

1. Delete the `tf = request.args.get("tf", "4h")` line, the `if tf not in TF_CFG: tf = "4h"` pair of lines, and `cfg = TF_CFG[tf]`.
2. After the `fetch_tickers` try/except, add:

```python
        funding = fetch_funding_map(main_ex)
        if not funding:
            yield sse("log", {"msg": "No bulk funding data — OI-flow health flags will be blank"})

        # Next curated macro print, for the entry verdict. A calendar failure
        # must not stop a scan; the verdict then simply ignores the calendar.
        ev_h, ev_title = None, ""
        try:
            refresh_calendar()
            with _lock:
                events = list(_cal_state["events"])
            ev_h, ev_title = next_mover(events, datetime.datetime.now(datetime.timezone.utc))
        except Exception as e:
            yield sse("log", {"msg": f"Calendar unavailable for the entry verdict: {e}"})
```

3. Change the "Scanning" log line to:

```python
        yield sse("log", {"msg": f"Scanning {len(pairs)} pairs on 4H + 1H · "
                                 f"{strictness} · {workers} workers"})
```

4. Replace the whole `scan_pair` function with:

```python
        def with_indicators(df):
            return pd.concat([df, compute_indicators(df)], axis=1).dropna()

        def scan_pair(sym):
            base = sym.split("/")[0]
            try:
                ex = worker_ex()
                raw = fetch_df_deep(ex, limiter, sym, "4h", TF_CFG["4h"]["bars"])
                if raw is None or len(raw) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return
                live_price = float(raw["close"].iloc[-1])

                # Drop the open candle BEFORE measuring volume: weekly_usd_volume
                # divides 7-day notional by the last 24h of notional, and a
                # still-forming bar in that denominator inflates the ratio.
                df4 = drop_forming_candle(raw, TF_CFG["4h"]["minutes"])
                if df4 is None or len(df4) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return

                # The volume gate runs on 4H alone, so a pair that fails it
                # never costs a 1H fetch.
                wk_vol = weekly_usd_volume(df4, vol24.get(sym), TF_CFG["4h"]["minutes"])
                if not only_syms and (wk_vol is None or wk_vol < min_vol):
                    with clock:
                        counter["vol_reject"] += 1
                    return

                df4 = with_indicators(df4)
                if len(df4) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return

                df1 = None
                raw1 = fetch_df_deep(ex, limiter, sym, "1h", TF_CFG["1h"]["bars"])
                if raw1 is not None and len(raw1) >= 80:
                    live_price = float(raw1["close"].iloc[-1])    # the newer tick
                    df1 = drop_forming_candle(raw1, TF_CFG["1h"]["minutes"])
                    df1 = with_indicators(df1) if df1 is not None else None

                with clock:
                    counter["analysed"] += 1

                sigs = (signals_for_tf(df4, "4h", rules, live_price, sym, wk_vol) +
                        signals_for_tf(df1, "1h", rules, live_price, sym, wk_vol))
                if not sigs:
                    return
                mover, flow, cerr = pair_context(ex, limiter, sym, live_price,
                                                 funding.get(sym), wk_vol)
                for row in build_rows(sigs, mover, flow, cerr, ev_h, ev_title):
                    q.put(("result", row))
            except Exception as e:
                with clock:
                    counter["errors"] += 1
                q.put(("log", {"msg": f"[{base}] skipped: {e}"}))
            finally:
                with clock:
                    counter["n"] += 1
                    n_now = counter["n"]
                q.put(("progress", {"current": n_now, "pair": sym}))
```

5. In `persist()`, change the first line of `meta` from `{"exchange": exid, "tf": tf, "strictness": strictness,` to:

```python
            meta = {"exchange": exid, "tfs": list(TF_CFG), "strictness": strictness,
```

- [ ] **Step 5: State, `/signals`, and deletions**

Replace the server-state block:

```python
# ── Server state ───────────────────────────────────────────────────────────────
_state = {"results": [], "ts": None, "meta": {}}
_lock  = threading.Lock()
_cal_state = {"events": [], "ts": None, "attempt": None, "meta": {},
              "jb": {"events": [], "ts": None, "attempt": None}}
```

Replace `save_state` and `load_state`:

```python
def save_state():
    try:
        with _lock:
            data = {"results": list(_state["results"]),
                    "ts": _state["ts"].isoformat() if _state["ts"] else None,
                    "meta": dict(_state["meta"])}
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[state] save failed: {e}")


def load_state():
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        meta = data.get("meta", {})
        if "tfs" not in meta:
            # Written before rows existed (flat signal list, single timeframe).
            # The page cannot draw it, so start empty rather than broken.
            print("[state] last scan is from an older version - ignored")
            return
        with _lock:
            _state["results"] = data.get("results", [])
            ts = data.get("ts")
            _state["ts"] = datetime.datetime.fromisoformat(ts) if ts else None
            _state["meta"] = meta
        print(f"[state] restored {len(_state['results'])} row(s) from last scan")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[state] load failed: {e}")
```

Replace `signals_route`:

```python
@app.route("/signals")
def signals_route():
    with _lock:
        return jsonify({"rows": list(_state["results"]),
                        "meta": dict(_state["meta"]),
                        "ts": _state["ts"].isoformat() if _state["ts"] else None})
```

Delete entirely:

- `watch_signals_route`, `watchlist_route`, `mover_signals_route`, `movers_route` (function, decorator and docstring each).
- `save_movers`, `load_movers`.
- The `MOVERS_STATE_FILE` assignment and the two comment lines above it.
- `load_movers()` in the `__main__` block.

In `classify_mover`, remove the `thin` parameter: signature becomes `def classify_mover(m):`, delete the docstring paragraph that starts `thin=True is the stricter test`, and replace the `if thin: ... else: ...` block with:

```python
    qualifies = (vm >= MOVER_VOL_MULT or rm >= MOVER_RANGE_MULT
                 or chg >= MOVER_MIN_CHG_PCT)
```

Leave `MOVER_THIN_MIN_WEEKLY_VOL` and the `index()` template dict alone; the page still references them until Task 6.

- [ ] **Step 6: Run all tests and check for leftovers**

Run: `python -m unittest discover -s tests -v`
Expected: all PASS.

Run: `grep -n "_movers_state\|MOVERS_STATE_FILE\|load_movers\|save_movers\|_state\[\"watch\"\]\|thin=" macd_divergence_screener_v5.py`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add tests macd_divergence_screener_v5.py
git commit -m "Scan 4H and 1H in one run and drop the Watchlist and Movers scans"
```

---

### Task 6: The page

All edits are inside the `HTML = r"""..."""` string unless stated. Line numbers are from before Task 1 and will have shifted; find blocks by the quoted anchors.

**Files:**
- Modify: `macd_divergence_screener_v5.py` (HTML string; `index()` template dict; `MOVER_THIN_MIN_WEEKLY_VOL`)
- Test: `tests/test_app.py` (append)

**Interfaces:**
- Consumes: `GET /signals` (`rows`, `meta`, `ts`), `/scan` SSE (`log`, `total`, `progress`, `result` = one row, `done` = `{hits, meta}`, `error`), row shape from Task 4, `entry` from Task 2, `mover.flags` and `flow.quadrant` / `flow.flags` from Task 3.
- Produces: page with nav pages `scan`, `calendar`, `learn`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app.py` before the `if __name__` block:

```python
import re  # noqa: E402


class Page(unittest.TestCase):
    def setUp(self):
        self.html = m.app.test_client().get("/").get_data(as_text=True)

    def test_three_pages(self):
        self.assertEqual(re.findall(r'class="nav-btn[^"]*" data-pg="(\w+)"', self.html),
                         ["scan", "calendar", "learn"])
        self.assertEqual(sorted(re.findall(r'<div id="pg-(\w+)"', self.html)),
                         ["calendar", "learn", "scan"])

    def test_no_unfilled_template_placeholders(self):
        self.assertEqual(re.findall(r"__[A-Z][A-Z_]+__", self.html), [])

    def test_removed_features_leave_no_code_behind(self):
        for gone in ("startWatch", "startMovers", "rtGo", "loadDash", "mvThin",
                     'data-val="loose"', 'data-val="1d"', "/watchlist", "/movers"):
            self.assertNotIn(gone, self.html, gone)

    def test_new_table_and_filters(self):
        for want in ('data-f="enter"', 'data-f="both"', "function evCell", "function ctxCell",
                     'id="scan-age"', 'id="dash-cal"'):
            self.assertIn(want, self.html, want)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m unittest tests.test_app -v`
Expected: `Page` tests FAIL (eight nav buttons, `startWatch` present, `data-f="enter"` missing).

- [ ] **Step 3: Markup — nav and removed pages**

Replace the eight `<button class="nav-btn...">` lines with:

```html
    <button class="nav-btn active" data-pg="scan">Scan</button>
    <button class="nav-btn" data-pg="calendar">Calendar</button>
    <button class="nav-btn" data-pg="learn">Method</button>
```

Delete these whole blocks, each from its opening `<div id="pg-...">` to the `</div>` just before the next `<div id="pg-...">`:

- `<div id="pg-dash" class="page active">`
- `<div id="pg-signals" class="page">`
- `<div id="pg-watch" class="page">`
- `<div id="pg-movers" class="page">`
- `<div id="pg-routine" class="page" ...>`

Change `<div id="pg-scan" class="page">` to `<div id="pg-scan" class="page active">`.

- [ ] **Step 4: Markup — the Scan page**

In the `<aside class="sb">`:

- Delete the whole `<div class="ss"><div class="fl">Timeframe</div> ... </div></div>` block (it contains `id="tfG"`).
- Delete the line `<button class="tbtn" data-val="loose">Loose</button>`.

Replace everything from `<div class="rbar">` through the closing `</div>` of `<div class="tw">` (the block that ends with `</table>` then `</div>`) with:

```html
    <div class="cal-strip" id="dash-cal" style="display:none;margin:10px 16px 0"></div>
    <div class="rbar">
      <span class="rtitle">Divergences</span>
      <span class="spill sbull" id="stBull">&#9650; <span class="num" id="cBull">0</span> bullish</span>
      <span class="spill sbear" id="stBear">&#9660; <span class="num" id="cBear">0</span> bearish</span>
      <span class="spill sfresh" id="stEnter">&#9889; <span class="num" id="cEnter">0</span> enter now</span>
      <div class="fbar">
        <button class="ftab active" data-f="live" onclick="setF(this)" title="Everything still worth watching: verdict Enter or Wait.">Live</button>
        <button class="ftab" data-f="enter" onclick="setF(this)" title="Only rows whose verdict is Enter.">Enter now</button>
        <button class="ftab" data-f="bull" onclick="setF(this)">Bullish</button>
        <button class="ftab" data-f="bear" onclick="setF(this)">Bearish</button>
        <button class="ftab" data-f="both" onclick="setF(this)" title="The same side diverging on 4H and 1H at once.">4H+1H</button>
        <button class="ftab" data-f="all" onclick="setF(this)" title="Includes Skip rows: stopped out, already ran, or too old.">All</button>
      </div>
    </div>
    <div id="scan-age" class="audit" style="margin:8px 16px 0"></div>
    <div id="scan-audit" class="audit" style="margin:4px 16px 8px"></div>
    <div class="tw">
      <div class="empty" id="emS">
        <span class="empty-icon">&#9672;</span><span>Run a scan to find divergences</span>
      </div>
      <table id="resT" style="display:none">
        <thead><tr>
          <th>Pair</th>
          <th title="A checklist, not a prediction. ENTER: second pivot is recent, price is within 2% of the signal, the stop is untouched, MACD has crossed its signal line since the pivot, and no market-mover print is due inside 24 hours. WAIT names the one thing still missing. SKIP: stopped out, already ran 1R, or too old. Judged at scan time on the 4H signal when there is one, else the 1H.">Entry</th>
          <th title="Regular bullish: price makes a lower low while the MACD line makes a higher low. Regular bearish: price makes a higher high while the MACD line makes a lower high. The tags show which timeframes diverge; 4H+1H agreeing sorts first.">Side &middot; TF</th>
          <th title="Not part of the verdict. Movers tags describe the last completed day against this coin's own 20-day baseline. OI tags describe 7-day open-interest flow and funding. Hover a tag for its definition.">Context</th>
          <th class="sortable sorted" data-sort="score" title="Default order: Enter, then Wait, then Skip; within each, 4H+1H first, then setup score. The score is how textbook the structure is, 0-100. Not a win rate, not backtested.">Setup score &#8595;</th>
          <th class="sortable" data-sort="vol" title="Rolling 7-day traded value in USD, measured from the 4H candles. The liquidity gate.">Weekly vol</th>
          <th class="sortable" data-sort="age" title="Time of pivot B, the bar that completed the divergence, on the row's main timeframe.">Signal date</th>
          <th title="Just past pivot B plus an ATR-sized buffer. Price trading through it means the divergence has failed. Risk = stop distance as a % of the entry reference.">Stop</th>
          <th title="Multiples of the stop distance. Anchored to the signal price while the setup is still near it; re-anchored to the live price once it has run.">Targets (1R / 2R / 3R)</th>
        </tr></thead>
        <tbody id="resB"></tbody>
      </table>
    </div>
```

- [ ] **Step 5: Markup — the Method page**

Inside `<div id="pg-learn" ...>`:

Replace the two sections titled `Watchlist &mdash; how to read OI flow` and `Movers &mdash; what a daily watchlist is actually screening for` (each a `<div class="dash-sec" style="margin-top:28px">` ... `</div>`) with:

```html
  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">Entry verdict &mdash; a checklist, not a prediction</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>ENTER means five things are true at scan time.</b> The second pivot is within the last 12 bars, price is within 2% of the signal price, the stop has not been touched, the MACD line has crossed its signal line since the pivot, and no market-mover print is due inside 24 hours. It does not mean the trade works.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>WAIT names the one thing missing.</b> A pullback to the signal price after price ran, the MACD cross, or a CPI / FOMC-class print that is too close. A divergence is listed as WAIT as soon as its second pivot is confirmed, so you see it before it is enterable.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>SKIP is history.</b> The stop was hit, the move already made 1R or more, or the second pivot is older than 12 bars. These rows only appear under the All filter.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>The verdict ages.</b> It uses the live price at the moment of the scan. Once the scan is more than an hour old, one full 1H bar has closed since, the verdicts dim and the page asks for a re-run.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>One row per coin and side.</b> When 4H and 1H diverge the same way the row shows both tags, sorts first, and takes its verdict, stop and targets from the 4H signal. Click a row for each timeframe's pivots and levels.</div></div>
    </div>
  </div>

  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">Context tags &mdash; what else is true about this coin</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Context never changes the verdict.</b> It is fetched only for coins that already have a divergence, and it is yours to weigh.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Movers tags describe the last completed day.</b> VOLUME SPIKE and RANGE EXPANSION are measured against the coin's own 20-day baseline, BIG MOVE is a 5%+ day, and the range tags say price is at or through its 7-day high or low. A bearish divergence AT RANGE HIGH is the textbook fade; the same divergence mid-range is weaker.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>OI tags describe 7-day positioning.</b> Price and open interest together give the quadrant: NEW LONGS, NEW SHORTS, SHORT COVERING, CAPITULATION. Funding refines it: SPOT-LED is the healthy version of new longs, CROWDED LONGS is leverage arriving late, SQUEEZE FUEL is shorts crowded and paying. No OI tag means open interest moved less than 5% or the contract is too new to judge.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>HIGH OI/VOL is a crowding warning, not a direction.</b> Open interest above about 1.5x average daily volume means positions cannot exit without moving the price. Size down.</div></div>
    </div>
  </div>
```

Then run `grep -n "1D\b\|Loose\|Scanner tab\|Signals tab\|Dashboard\|Watchlist tab\|Movers tab\|Routine" macd_divergence_screener_v5.py` restricted by eye to the `pg-learn` block and the sidebar hints. For each hit inside the HTML string: reword to the current names (the only tabs are Scan, Calendar, Method; the timeframes are 4H and 1H; the presets are Balanced and Strict). If the Method rules mention pivots needing "5 bars" on both sides, change the wording to "5 bars before it and 3 closed bars after it".

- [ ] **Step 6: CSS**

Delete these rules (whole `selector{...}` lines or blocks):

- `#pg-dash.active,#pg-signals.active,#pg-watch.active{...}` and `#pg-dash,#pg-signals,#pg-watch{...}`
- `#pg-movers.active{...}`, `#pg-movers{...}`
- Every rule whose selector starts with `.rt-`, the `@keyframes rtIn`, `rtBeat`, `rtFlowDot`, `rtTimer`, `rtPanel`, `rtFlowDotV` blocks, and the `@media (max-width:860px)` and `@media (prefers-reduced-motion:reduce)` blocks that contain only `.rt-` rules
- `.mv-thin`, `.mv-thin input`, `.mv-sep td`, `.thin-badge`, `.mrank`, `.mrank.top`, `.mpos`, `.mpos i`, `.mlv`, `.mlv small`
- `.kpi-row`, `.kpi`, `.kpi-lbl`, `.kpi-val`, `.kpi-sub`, `.ready-grid`, `.sig-card`, `.sc-timing`, `.sc-body`, `.sc-header`, `.sc-pair`, `.sc-meta`, `.sml`, `.smv.bull` (and any other `.smv` rule), `.qbar-wrap`, `.qbar-lbl`, `.qbar-bg`, `.qbar-fill`, `.sig-list`, `.srow`, `.srow:hover`, `.srow.dim`, `.srow-pair`, `.tgt-row`

Keep `.wq*`, `.wflag`, `.wf-*`, `.mf-*`, `.wbar*`, `.wprog`, `.sig-hdr`, `.sig-ttl`, `.sig-sub`, `.sig-filters`, `.sig-ftab*`, `.dash-sec*`, `.no-sig`, `.audit*`, `.rule*`: the Calendar and Method pages and the new badges use them.

Add next to the `.st-invalidated{...}` rule:

```css
.ev{display:inline-flex;flex-direction:column;gap:1px;padding:5px 9px;border-radius:5px;font-family:var(--mono);min-width:150px;max-width:260px}
.ev .tsub{white-space:normal;line-height:1.35}
.ev-enter{background:#0D2E1F;color:#34D399;border:1px solid #1a4020}
.ev-wait{background:var(--fresh-bg);color:var(--fresh);border:1px solid #4a3500}
.ev-skip{background:var(--sf3);color:var(--tx3);border:1px solid var(--bd2)}
table.stale .ev{opacity:.4}
tbody tr.main-row{cursor:pointer}
tr.det td{background:var(--sf);font-family:var(--mono);font-size:10px;color:var(--tx2);line-height:1.8;padding:10px 16px}
tr.det b{color:var(--tx)}
.tfb{display:inline-block;margin-left:4px;padding:2px 5px;border-radius:3px;font-family:var(--mono);font-size:9px;font-weight:700;background:var(--sf3);color:var(--tx2);border:1px solid var(--bd2)}
```

- [ ] **Step 7: JavaScript — delete**

Inside `<script>`:

- Delete `var cTf = '4h';`.
- Delete `function freshTag(d){...}` and `function sbadge(d){...}` (keep the `SC` object; `detailRow` uses it).
- Delete the whole `/* ── Dashboard ── */` section: `loadDash`, `renderDash`, `mkCard`, `mkRow`.
- Delete the whole `/* ── Signals page ── */` section: `_sf`, `_sigs`, `loadSignals`, `setSF`, `renderSigs`.
- In the `/* ── Watchlist (OI flow) ── */` section keep `WQ`, `WFLAG`, `TH`, `TIP`, `tip` and delete the rest: the `var _wf = 'all', _wrows = [], wes = null;` line, `wMatches`, `setWF`, `fmtPct`, `watchAudit`, `renderWatch`, `loadWatch`, `startWatch`.
- In the `/* ── Movers ── */` section keep `MFLAG` and delete the rest: the `var _mf = 'all', _mrows = [], mes = null;` line, `mMatches`, `setMF`, `fmtX`, `moversAudit`, `mvRow`, `renderMovers`, `mvThinPref`, `loadMovers`, `startMovers`.
- Delete the whole `/* ── Routine flow ── */` section including the `document.addEventListener('keydown', ...)` handler that follows `rtStop`.
- In the `TH` object delete `thinM:__MOVER_THIN_M__, ` (keep the rest of the line).
- In `STR_HINTS` delete the `loose:` line.

- [ ] **Step 8: JavaScript — replace**

Replace `function matches(d, f){...}` with:

```js
var EV = {
  enter: {lbl:'&#10003; ENTER', cls:'ev-enter'},
  wait:  {lbl:'&#9203; WAIT',   cls:'ev-wait'},
  skip:  {lbl:'&#10005; SKIP',  cls:'ev-skip'}
};
var EV_RANK = {enter:0, wait:1, skip:2};
function verdict(r){ return (r.entry && r.entry.verdict) || 'skip'; }
function primary(r){ return r.signals[r.tfs[0]]; }

function matches(r, f){
  var v = verdict(r);
  if(f === 'all')   return true;
  if(f === 'live')  return v !== 'skip';
  if(f === 'enter') return v === 'enter';
  if(f === 'bull')  return r.side === 'bullish' && v !== 'skip';
  if(f === 'bear')  return r.side === 'bearish' && v !== 'skip';
  if(f === 'both')  return r.tfs.length > 1 && v !== 'skip';
  return true;
}
function evCell(r){
  var e = r.entry || {verdict:'skip', reason:''}, c = EV[e.verdict] || EV.skip;
  /* the reason can carry a calendar event title from an outside feed */
  var txt = calEsc(e.reason) + (e.level != null ? ' ' + fmtPx(e.level) : '');
  return '<div class="ev ' + c.cls + '"><span class="tlbl">' + c.lbl + '</span>' +
         '<span class="tsub">' + txt + '</span></div>';
}
function ctxCell(r){
  if(r.context_error) return '<span class="ptm" title="' + calEsc(r.context_error) + '">context unavailable</span>';
  var h = [];
  ((r.mover || {}).flags || []).forEach(function(f){
    h.push('<span class="wflag ' + (MFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>');
  });
  if(r.flow){
    h.push('<span class="' + (WQ[r.flow.quadrant] || {cls:'wq'}).cls + '"' + tip(r.flow.quadrant) + '>' + r.flow.quadrant + '</span>');
    (r.flow.flags || []).forEach(function(f){
      h.push('<span class="wflag ' + (WFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>');
    });
  }
  return h.length ? '<div style="display:flex;flex-wrap:wrap;gap:4px;max-width:260px">' + h.join('') + '</div>'
                  : '<span class="ptm">&mdash;</span>';
}
function tfLabel(tf){ return (TFM[tf] && TFM[tf].label) || tf.toUpperCase(); }
```

Replace `function showPg(name){...}` with:

```js
function showPg(name){
  document.querySelectorAll('.page').forEach(function(p){ p.classList.remove('active'); });
  document.getElementById('pg-' + name).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(function(b){ b.classList.toggle('active', b.dataset.pg === name); });
  if(name === 'calendar') loadCalendar();
}
```

Change the control-group listener list from `['tfG','strG','spdG']` to `['strG','spdG']`.

Replace `updStats` and `resetStats` with:

```js
function updStats(side, isEnter){
  if(side === 'bullish'){ cb++; bump(document.getElementById('cBull'), cb); document.getElementById('stBull').classList.add('vis'); }
  else { cr++; bump(document.getElementById('cBear'), cr); document.getElementById('stBear').classList.add('vis'); }
  if(isEnter){ cf++; bump(document.getElementById('cEnter'), cf); document.getElementById('stEnter').classList.add('vis'); }
}
function resetStats(){
  cb = cr = cf = 0;
  ['stBull','stBear','stEnter'].forEach(function(i){ document.getElementById(i).classList.remove('vis'); });
  ['cBull','cBear','cEnter'].forEach(function(i){ document.getElementById(i).textContent = '0'; });
}
```

Replace `function buildRow(d){...}` and `function renderAll(results){...}` with:

```js
function tgtHtml(d){
  var tgt = d.targets || {}, fromCur = d.targets_from === 'current';
  return '<div style="font-family:var(--mono);font-size:8px;color:' + (fromCur ? 'var(--fresh)' : 'var(--tx3)') + ';margin-bottom:2px">' +
    (fromCur ? 'FROM CURRENT PRICE ' + fmtPx(d.entry_ref_price) : 'FROM SIGNAL PRICE') + '</div>' +
    '<span style="color:var(--tx3)">' + fmtPx(tgt['1r']) + '</span>' +
    ' <span style="color:var(--tx3)">/</span> <span style="color:var(--tx2)">' + fmtPx(tgt['2r']) + '</span>' +
    ' <span style="color:var(--tx3)">/</span> <span style="color:var(--bull);font-weight:600">' + fmtPx(tgt['3r']) + '</span>';
}
function detailRow(r){
  var tr = document.createElement('tr');
  tr.className = 'det';
  var h = r.tfs.map(function(tf){
    var s = r.signals[tf], c = SC[s.status] || SC.active;
    return '<div><b>' + tfLabel(tf) + '</b> &nbsp;A ' + s.pivot1_time + ' @ ' + fmtPx(s.pivot1_price) +
      ' &rarr; B ' + s.pivot2_time + ' @ ' + fmtPx(s.pivot2_price) +
      ' &nbsp;&middot;&nbsp; MACD ' + s.pivot1_macd + ' &rarr; ' + s.pivot2_macd +
      ' &nbsp;&middot;&nbsp; ' + c.lbl + ' (' + (s.pct_chg >= 0 ? '+' : '') + s.pct_chg + '% in signal direction)' +
      ' &nbsp;&middot;&nbsp; ' + (s.macd_cross ? 'MACD crossed ' + s.macd_cross_at : 'no MACD cross yet') +
      ' &nbsp;&middot;&nbsp; score ' + s.score +
      ' &nbsp;&middot;&nbsp; stop ' + fmtPx(s.stop_price) + ' (risk ' + s.risk_pct + '%)' +
      ' &nbsp;&middot;&nbsp; 1R ' + fmtPx((s.targets || {})['1r']) + ' / 2R ' + fmtPx((s.targets || {})['2r']) +
      ' / 3R ' + fmtPx((s.targets || {})['3r']) + '</div>';
  }).join('');
  tr.innerHTML = '<td colspan="9">' + h + '</td>';
  return tr;
}
function buildRow(r){
  var bull = r.side === 'bullish', p = primary(r), tf = r.tfs[0];
  var tr = document.createElement('tr');
  tr.className = 'main-row ' + (bull ? 'rb' : 'rr') + (verdict(r) === 'skip' ? ' dim' : '');
  var tfs = r.tfs.map(function(t){ return '<span class="tfb">' + tfLabel(t) + '</span>'; }).join('');
  tr.innerHTML =
    '<td><div class="pb"><span class="pb-pair">' + r.base + '</span><span class="pb-q">USDT PERP</span></div></td>' +
    '<td>' + evCell(r) + '</td>' +
    '<td><span class="bdg ' + (bull ? 'bb' : 'br') + '">' + (bull ? 'BULLISH' : 'BEARISH') + '</span>' + tfs + '</td>' +
    '<td>' + ctxCell(r) + '</td>' +
    '<td>' + qcell(r.score || 0) + '</td>' +
    '<td>' + volCell(r) + '</td>' +
    '<td><div class="ptm" style="color:var(--tx);font-size:11px">' + p.pivot2_time + '</div><div class="ptm">' + tfLabel(tf) + ' &middot; ' + b2t(p.bars_ago, tf) + '</div></td>' +
    '<td><span style="font-family:var(--mono);font-size:11px;color:var(--bear)">' + fmtPx(p.stop_price) + '</span><div style="font-family:var(--mono);font-size:9px;color:var(--tx3)">risk ' + p.risk_pct + '%</div></td>' +
    '<td style="font-family:var(--mono);font-size:10px">' + tgtHtml(p) + '</td>';
  tr.addEventListener('click', function(){
    var n = tr.nextSibling;
    if(n && n.className === 'det'){ n.parentNode.removeChild(n); return; }
    tr.parentNode.insertBefore(detailRow(r), tr.nextSibling);
  });
  return tr;
}
function ageMin(r){ var tf = r.tfs[0]; return primary(r).bars_ago * ((TFM[tf] && TFM[tf].minutes) || 240); }

function renderAll(results){
  var rows = (results || []).filter(function(d){ return matches(d, af); });
  rows.sort(function(a, b){
    if(sortKey === 'vol') return (b.weekly_volume || 0) - (a.weekly_volume || 0);
    if(sortKey === 'age') return ageMin(a) - ageMin(b);
    return (EV_RANK[verdict(a)] - EV_RANK[verdict(b)]) ||
           (b.tfs.length - a.tfs.length) || (b.score - a.score);
  });
  var tb = document.getElementById('resB'); tb.innerHTML = '';
  var st = rows.length > 1 ? Math.min(60, 800 / rows.length) : 0;
  rows.forEach(function(d, i){
    var tr = buildRow(d);
    tr.style.animationDelay = (i * st).toFixed(0) + 'ms';
    tb.appendChild(tr);
  });
  if(rows.length === 0){
    document.getElementById('resT').style.display = 'none';
    var em = document.getElementById('emS');
    em.style.display = 'flex';
    em.innerHTML = '<span class="empty-icon">&#9676;</span><span>No divergences match this filter</span>';
  } else {
    document.getElementById('emS').style.display = 'none';
    document.getElementById('resT').style.display = 'table';
  }
}

/* The verdict is only as fresh as the scan. Past one closed 1H bar it is dimmed. */
var scanTs = null;
function updAge(){
  var el = document.getElementById('scan-age');
  if(!scanTs){ el.textContent = ''; el.className = 'audit'; return; }
  var min = Math.max(0, Math.round((Date.now() - scanTs) / 60000)), stale = min > 60;
  el.className = 'audit' + (stale ? ' warn' : '');
  el.innerHTML = 'Entry verdicts as of ' + min + ' min ago' +
    (stale ? ' &mdash; more than one 1H bar has closed since. Re-run the scan before acting on them.' : '');
  document.getElementById('resT').classList.toggle('stale', stale);
}
setInterval(updAge, 60000);

function loadLast(){
  fetch('/signals').then(function(r){ return r.json(); }).then(function(d){
    var rows = d.rows || [];
    if(!d.ts) return;                       /* never scanned: leave the welcome panel */
    pendR = rows; hasScanned = true;
    scanTs = new Date(d.ts).getTime();
    resetStats();
    rows.forEach(function(r){ if(verdict(r) !== 'skip') updStats(r.side, verdict(r) === 'enter'); });
    document.getElementById('scan-audit').innerHTML = auditLine(d.meta);
    renderAll(pendR); updAge();
  }).catch(function(){});
}
```

In `startScan`:

- Keep the first line, `showPg('scan');`.
- Delete the line `var tf   = document.querySelector('#tfG .tbtn.active').dataset.val;` and the line `cTf = tf;`.
- Replace the URL lines with:

```js
  var url = '/scan?exchange=' + exch + '&strictness=' + str +
            '&speed=' + spd + '&min_weekly_vol=' + vol + '&max_pairs=' + mp;
```

- After `showLoader(); resetBg(); pendR = []; seenSig = {}; hasScanned = true;` add:

```js
  scanTs = null; updAge();
  document.getElementById('scan-audit').innerHTML = '';
```

- Replace the `result` listener with:

```js
  es.addEventListener('result', function(e){
    var d = JSON.parse(e.data);
    var k = d.pair + '|' + d.side;
    if(seenSig[k]) return;            // duplicate guard, see es.onerror
    seenSig[k] = 1;
    if(verdict(d) !== 'skip') updStats(d.side, verdict(d) === 'enter');
    pendR.push(d);
    pushBg(d.pair, true);
  });
```

- In the `done` listener, replace `setStatus('Done - ' + d.hits + ' signal' + ...)` and the `pushLog('Scan complete ...')` line with:

```js
    setStatus('Done - ' + d.hits + ' row' + (d.hits !== 1 ? 's' : ''), false);
    var m = d.meta || {};
    pushLog('Scan complete - ' + d.hits + ' divergence row(s) from ' + (m.scanned || 0) + ' pairs analysed', 'dn');
    scanTs = Date.now(); updAge();
    document.getElementById('scan-audit').innerHTML = auditLine(m);
```

  (the existing `var m = d.meta || {};` line is replaced by the one above; do not declare `m` twice).

- In `es.onerror`, inside `if(pendR.length){...}` add `scanTs = Date.now(); updAge();` before `dissolve();`.

In `renderWelcome`, replace the `subs` array with:

```js
    var subs = ['Ready to find divergences.',
                '4H and 1H, regular divergences only.',
                'Every liquid perp, one scan, one table.',
                'Set the volume floor and hit Run.'];
```

Replace the `DOMContentLoaded` handler with:

```js
document.addEventListener('DOMContentLoaded', function(){
  renderWelcome();
  loadLast();
  loadCalendar();
  var pg = (location.hash || '').slice(1);
  if(pg && document.getElementById('pg-' + pg)) showPg(pg);   /* deep link, e.g. /#calendar */
});
```

- [ ] **Step 9: Python side of the page**

In `index()`, delete the `"__MOVER_THIN_M__": ...` entry from `tmpl`.

Delete the `MOVER_THIN_MIN_WEEKLY_VOL` assignment and the seven-line `# Thin band (opt-in, ...` comment above it.

- [ ] **Step 10: Run tests and check the script parses**

Run: `python -m unittest discover -s tests -v`
Expected: all PASS.

Run:

```bash
python -c "import re,macd_divergence_screener_v5 as m; h=m.app.test_client().get('/').get_data(as_text=True); open('page_check.js','w',encoding='utf-8').write(re.search(r'<script>(.*)</script>', h, re.S).group(1))" && node --check page_check.js && rm page_check.js
```

Expected: no output from `node --check` (a syntax error prints the line).

Run: `grep -n "cTf\|freshTag\|sbadge\|mkRow\|mkCard\|rtStop\|_sigs\|_wrows\|_mrows\|MOVER_THIN\|sig-list\|kpi-" macd_divergence_screener_v5.py`
Expected: no output.

- [ ] **Step 11: Look at it**

Run: `python macd_divergence_screener_v5.py`, open `http://localhost:5099`.

Check: three tabs; Scan page shows settings left and the welcome panel right; Calendar and Method open and scroll; `/#calendar` deep link works; the calendar strip appears above the filter bar; no errors in the browser console. Do not run a scan yet (Task 8).

- [ ] **Step 12: Commit**

```bash
git add tests/test_app.py macd_divergence_screener_v5.py
git commit -m "Rebuild the page as one Scan tab with entry verdicts and context tags"
```

---

### Task 7: README and module docstring

**Files:**
- Modify: `README.md`
- Modify: `macd_divergence_screener_v5.py` (module docstring only)

**Interfaces:** none.

- [ ] **Step 1: README intro and "What it does"**

Replace the first paragraph with:

```markdown
A self-contained Flask app that scans linear USDT perpetuals on OKX, Binance,
and Bybit for **regular MACD divergences** (bullish and bearish) on the **4H
and 1H** timeframes in one run. Each result row says whether the setup is
enterable right now, and carries daily-movers and open-interest context for
that coin. A news calendar covers the scheduled macro prints that move
crypto. Everything runs from one file: `macd_divergence_screener_v5.py`
serves both the API and a browser dashboard.
```

In "What it does":

- In **Divergence detection**, after "matched to a MACD pivot within a few bars," insert: `confirmed once 3 closed bars follow it (5 bars before, 3 after),`. After the sentence ending "filter out micro-divergences." add: `The two pivots are at most 60 bars apart, and a divergence is dropped once its second pivot is 30 bars old on 4H or 24 bars old on 1H.`
- Replace the **Strictness presets** bullet with:

```markdown
- **Strictness presets**: `balanced` (default) and `strict`. Strict adds the
  zero-line condition and a larger minimum divergence.
```

- Replace the **OI flow watchlist** and **Daily movers** bullets with:

```markdown
- **One row per coin and side**: when 4H and 1H diverge the same way they
  share a row, which sorts first and takes its stop and targets from the 4H
  signal.
- **Entry verdict**: every row is `ENTER`, `WAIT` or `SKIP` with the reason.
  `ENTER` means the second pivot is within 12 bars, price is within 2% of the
  signal price, the stop is untouched, MACD has crossed its signal line since
  the pivot, and no market-mover print is due inside 24 hours. It is a
  checklist judged at scan time, not a prediction; the page dims verdicts
  once the scan is over an hour old.
- **Context tags**: for coins that have a divergence, the scan also reads the
  last completed day against the coin's 20-day baseline (volume spike, range
  expansion, at the 7-day high or low) and 7-day open-interest flow with
  funding (new longs, crowded longs, squeeze fuel, spot-led). Tags never
  change the verdict.
```

- [ ] **Step 2: README API and parameters**

Replace the API table with:

```markdown
| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/scan` | GET (SSE) | Run the 4H + 1H divergence scan |
| `/signals` | GET | Last cached `/scan` result (`rows`, `meta`, `ts`) |
| `/universe` | GET | Size of the tradable universe at a given volume floor (no scan) |
| `/calendar` | GET | Upcoming economic events, cached; `refresh=1` forces a re-fetch (at most every 5 min) |
```

Change the sentence above it to: `The scan endpoint streams progress via SSE (`log`, `progress`, `result`, `error`, `done` events) and caches its last completed result, retrievable without re-scanning.`

In the query-parameter list: delete the `tf` and `thin` bullets, change the strictness bullet to `` `strictness` — `balanced` (default) or `strict` ``, and drop "(`/scan` only)" from the `symbols` bullet.

- [ ] **Step 3: README — replace "Daily routine"**

Replace the whole `## Daily routine` section (through the line before `## Disclaimer`) with:

```markdown
## Reading a row

1. **Check the calendar strip first.** If a market mover is inside 24 hours,
   every otherwise-ready row reads `WAIT` and names the event.
2. **Read the audit line.** A short list from an unfinished scan is not a
   quiet market.
3. **Start with `ENTER` rows, 4H+1H first.** Click a row for both
   timeframes' pivots and check the line on your chart before anything else.
4. **`WAIT` rows are the watchlist.** The reason says what to wait for: the
   MACD cross, a pullback to the signal price, or the macro print.
5. **Weigh the context tags.** A bearish divergence `AT RANGE HIGH` with
   `CROWDED LONGS` is a stronger fade than the same divergence mid-range.
   `SQUEEZE FUEL` under a bearish row is a warning. `HIGH OI/VOL` means size
   down.
6. **Re-run before acting on an old table.** Verdicts use the price at scan
   time.

The *Specific coins* box scans only the tickers you type and skips the
volume floor, so a thin name can still be checked on demand. Use the
exchange with the deepest book (Binance for most pairs).
```

In the "News calendar" section, change "**The Dashboard strip**" to "**The Scan-page strip**".

- [ ] **Step 4: Module docstring**

Change the first docstring line to:

```
"""MACD Divergence Screener v5 — 4H + 1H regular divergences, one scan, one table
```

Insert directly under it, before `WHAT CHANGED FROM v3.3`:

```
WHAT CHANGED IN v5
──────────────────
One scan instead of three. /scan reads 4H and 1H for every pair, groups the
signals into one row per coin and side, and attaches an entry verdict
(entry_verdict: a checklist, not a prediction). The OI-flow and daily-movers
scans no longer exist as lists; their classifiers run only for coins that
have a divergence and show up as context tags. 1D and the Loose preset are
gone. A pivot is confirmed 3 closed bars after it instead of 5, the two
pivots may be at most 60 bars apart, and Balanced needs the MACD pivots to
differ by 15% of the leg's range. Those last two numbers are a first guess,
not tuned against missed examples.
```

In the same docstring, in the paragraph beginning `Strictness is selectable.`, delete the final sentence `LOOSE drops the MACD-pivot-alignment and clean-line requirements — more signals, more junk.`

- [ ] **Step 5: Run tests and commit**

Run: `python -m unittest discover -s tests -v`
Expected: all PASS.

```bash
git add README.md macd_divergence_screener_v5.py
git commit -m "Document the one-scan screener"
```

---

### Task 8: Live check and before/after signal counts

Needs network access to Binance. Nothing in this task is committed.

**Files:** none.

**Interfaces:** consumes everything above.

- [ ] **Step 1: Count signals under the old and the new geometry on the same candles**

Run from the repo root:

```bash
python - <<'EOF'
import pandas as pd
import macd_divergence_screener_v5 as m

ex = m.make_exchange("binance")
mk = ex.load_markets()
tk = ex.fetch_tickers()
perps = [s for s in mk if s.endswith("/USDT:USDT") and mk[s].get("active")
         and s.split("/")[0] not in m.STABLE_BASES and not m.is_tradfi(mk[s])]
perps = sorted(perps, key=lambda s: m.usd_volume(tk.get(s)) or 0, reverse=True)[:40]
lim = m.RateLimiter(8)

OLD_RULES = dict(m.STRICTNESS["balanced"], min_macd_frac=0.08)
tot = {}
for tf in m.TF_CFG:
    new = m.TF_CFG[tf]
    old = dict(new, pivot_rb=5, max_dist=120, max_age=60)
    for name, cfg, rules in (("old", old, OLD_RULES), ("new", new, m.STRICTNESS["balanced"])):
        tot[(tf, name)] = [0, 0]
    for s in perps:
        df = m.drop_forming_candle(m.fetch_df_deep(ex, lim, s, tf, new["bars"]), new["minutes"])
        if df is None or len(df) < 80:
            continue
        df = pd.concat([df, m.compute_indicators(df)], axis=1).dropna()
        for name, cfg, rules in (("old", old, OLD_RULES), ("new", new, m.STRICTNESS["balanced"])):
            sigs = m.detect_divergences(df, cfg, rules)
            tot[(tf, name)][0] += len(sigs)
            tot[(tf, name)][1] += sum(1 for x in sigs if x["fresh"])
for (tf, name), (n, fresh) in tot.items():
    print(f"{tf} {name}: {n} divergences, {fresh} fresh  (top 40 Binance perps)")
EOF
```

Expected: four lines. Record them in the hand-off message. `new` listing fewer total divergences than `old` is the intent (shorter window, larger minimum); the fresh counts show what the 3-bar confirmation adds.

- [ ] **Step 2: Run a real scan in the browser**

Run: `python macd_divergence_screener_v5.py`, open `http://localhost:5099`, choose Binance, set Max pairs to 60, press Run Scan.

Check each of these and note any that fail:

1. Progress runs to 100% and the audit line reads `N of 60 candidate pairs analysed` with the other buckets adding up.
2. Rows appear with an Entry badge and reason, Side and timeframe tags, and Context tags or a dash.
3. Default order is ENTER, then WAIT; SKIP rows appear only under All.
4. The Enter now, Bullish, Bearish and 4H+1H filters each show the expected subset.
5. Clicking a row opens the detail line with each timeframe's two pivots; clicking again closes it.
6. "Entry verdicts as of 0 min ago" shows above the table.
7. Reload the page: the same rows come back from `/signals`.
8. Pick two rows and compare pivot A and B times and prices against the exchange chart with MACD (12, 26, 9). The line should be visible on the MACD line.

- [ ] **Step 3: Report**

Report the four count lines from Step 1, the result of each check in Step 2, and for check 8 the two coins used and whether the divergence was visible. If a divergence was not visible, do not retune here: record the coin, timeframe and pivot times so they can become a regression test.
