# MACD Wave Divergence (spec v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the price-pivot divergence detector with the MACD crossover-wave detector of spec v2, and add the quality grade, regime tag, lower-timeframe trigger, chart preview and a backtest mode.

**Architecture:** Everything stays in `macd_divergence_screener_v5.py`. One pure function, `detect_wave_divergence(df, side, preset)`, finds divergences from MACD/signal crossover waves; the live scan and the backtest both call it. Pure helpers turn a detection into a signal record, score it and judge it; the scan route only fetches data and wires them together. The backtest is a command-line mode (`--backtest`) that replays the same candles bar by bar.

**Tech Stack:** Python 3.9+, Flask, ccxt, pandas, numpy, matplotlib (new, chart preview only), unittest.

**Spec:** `docs/superpowers/specs/2026-11-19-macd-wave-divergence-v2.md`

## Global Constraints

- One Python file: `macd_divergence_screener_v5.py`. Do not rename it (the tests and the README import it by this name) and do not split it.
- Only new dependency: `matplotlib`. Nothing else.
- MACD 12/26/9 on close, ATR 14 (Wilder), EMA 200, RSI 14 on BTC 4H.
- History: 720 bars of 4H, 500 bars of 1H, 200 bars of 1D.
- Regular divergences only. The histogram is never used for detection.
- Universe, Movers tab and Calendar tab are unchanged.
- Tests are `unittest`, run from the repo root: `python -m unittest discover -s tests -v`.
- Commit messages and PR text carry no Claude / AI attribution lines.
- Work on a branch: `git checkout -b feature/macd-wave-divergence`.

## Interpretations of the spec

The spec leaves these open. The plan fixes them as follows; every one is a named constant or a short function, so it is cheap to change.

1. **Whipsaw window.** "Within 3 bars" means the cross bar and the next two. The gap is `|MACD - signal|` on the side of the cross; the "50-bar MACD range" is max minus min of the MACD line over the 50 bars ending at the cross bar. A cross that fails is dropped, which merges the wave it would have started.
2. **Short waves.** A wave shorter than 3 bars is removed by deleting the two crosses around it, so the waves on either side join.
3. **Wave bounds.** A wave covers the bars from its opening cross up to the bar before its closing cross. The price extreme is searched from the opening cross to one bar after the closing cross ("2 bars after the wave end"). At the signal bar itself that last bar does not exist yet, so price low B (and the stop) can move down once, one bar later.
4. **"Yes (early)".** The rest of the spec says only completed candles are used, but "Yes (early)" needs a signal bar that is still forming. The scan therefore runs the detector a second time on the candles *including* the open one and keeps only a signal whose signal bar is that open candle. Such a row is provisional: it answers "Yes (early)" if the lower-timeframe trigger has fired, otherwise "Not yet" (wait for the bar to close). It can disappear on the next scan.
5. **Trigger "fired first".** The trigger looks at lower-timeframe bars from the signal bar's open onward. It counts as "first" (and becomes the entry reference) when the trigger bar closed before the signal bar closed.
6. **Higher-timeframe agreement and regime** are judged as of the signal bar's close, not as of now, so the live scan and the backtest give the same grade to the same signal. "Rising" = the last completed higher-timeframe MACD value is above the one before it.
7. **OI "into B"** = change in open interest from the start of wave B to price extreme B. Daily OI points are used for 4H signals and hourly points for 1H signals. When the history does not cover both moments the component scores 0 and no penalty applies.
8. **Status and R** stay measured from the signal price (existing `evaluate_status`), even when the entry reference is the trigger price.
9. **Backtest** enters at the close of the bar on which the signal first becomes visible, checks the stop before the targets inside one bar, and cannot score open interest or use the lower-timeframe trigger (no bar-by-bar history for either). The report says so.
10. **Chart data** is kept in memory for the last scan only. After a restart the table is restored from disk but charts need a fresh scan.
11. **Weekly volume** is not in the spec's column list, so it moves from a table column to the detail row.

## File Structure

| File | Change |
|---|---|
| `macd_divergence_screener_v5.py` | Detector, signal record, score, verdict, context, scan wiring, chart, backtest, dashboard table |
| `tests/helpers.py` | New. `wave_df()` builds candles with hand-made MACD waves |
| `tests/test_waves.py` | New. Crosses, waves, detection |
| `tests/test_divergence.py` | Rewritten. Pivots (kept), config, `signals_for_tf`, score, regime |
| `tests/test_rows.py` | Entry-verdict and row tests updated |
| `tests/test_context.py` | New. Higher-timeframe, trigger, OI, `add_context` |
| `tests/test_app.py` | State version, `usdt_perps`, chart route, page checks |
| `tests/test_backtest.py` | New. Outcome, replay, summary |
| `README.md` | Detection, score, verdict, requirements, backtest |

---

### Task 1: Wave detector

**Files:**
- Modify: `macd_divergence_screener_v5.py` (constants after `ATR_PERIOD` ~L111; `compute_indicators` ~L324; new functions after `find_pivots`/`swing_pivots` ~L375)
- Create: `tests/helpers.py`, `tests/test_waves.py`

**Interfaces:**
- Produces: `rsi(close, p=RSI_PERIOD) -> pd.Series`; `compute_indicators(df)` gains an `ema200` column; `macd_crosses(macd, signal) -> list[(bar:int, direction:+1|-1)]`; `detect_wave_divergence(df, side, preset) -> list[dict]` with keys `side, a_m, b_m, a_p, b_p, a_macd, b_macd, a_price, b_price, a_start, b_start, signal_bar, bump_bar, bump_frac, macd_frac, price_atr, zero_ok, triple, vol_dry, wick, crosses`. `df` needs columns `open high low close volume macd signal atr`. `preset` needs keys `min_price_pct, atr_frac, min_macd_frac, zero_line`.
- Produces (tests): `tests.helpers.wave_df(lobes, side="bullish", freq="1h", offset=0.0)`, `tests.helpers.W`, `tests.helpers.LONG`.

- [ ] **Step 1: Write the test helper**

Create `tests/helpers.py`:

```python
"""Synthetic candles whose MACD line is built by hand, one lobe per wave."""
import numpy as np
import pandas as pd


def wave_df(lobes, side="bullish", freq="1h", offset=0.0):
    """lobes: (bars, MACD amplitude, price depth), alternating, the first one a bump.

    For side="bullish" the odd lobes are troughs below a flat signal line and
    price dips by `depth` inside them. side="bearish" is the same picture
    upside down. `offset` lifts MACD and signal together (zero-line tests).
    """
    sgn = 1 if side == "bullish" else -1
    macd, ext = [], []
    for k, (n, amp, depth) in enumerate(lobes):
        shape = np.sin(np.pi * (np.arange(n) + 0.5) / n)
        bump = k % 2 == 0
        macd.append((amp if bump else -amp) * shape)
        ext.append(np.zeros(n) if bump else -depth * shape)
    macd = sgn * np.concatenate(macd) + offset
    ext = 100 + sgn * np.concatenate(ext)
    low, high = (ext, ext + 1) if side == "bullish" else (ext - 1, ext)
    idx = pd.date_range("2026-01-01", periods=len(macd), freq=freq)
    return pd.DataFrame({"open": low + .5, "high": high, "low": low, "close": low + .5,
                         "volume": 1.0, "macd": macd, "signal": offset, "atr": 1.0,
                         "ema200": 100.0}, index=idx)


# Trough A (MACD -2, price -5), bump, trough B (MACD -1, price -8), closing cross at bar 44.
W = [(10, 1, 0), (12, 2.0, 5), (10, 1.2, 0), (12, 1.0, 8), (8, 1, 0)]
# The same picture after 60 quiet bars: 102 bars, closing cross at bar 94.
LONG = [(60, 1, 0)] + W[1:]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_waves.py`:

```python
"""Tests for MACD crossover waves and wave divergence.

Run with:  python -m unittest tests.test_waves -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import W, wave_df  # noqa: E402

RULES = {"min_price_pct": 0.25, "atr_frac": 0.20, "min_macd_frac": 0.15, "zero_line": False}
CROSSES = [(10, -1), (22, 1), (32, -1), (44, 1)]


def crosses(df):
    return m.macd_crosses(df["macd"].values, df["signal"].values)


def detect(lobes, side="bullish", rules=RULES, **kw):
    return m.detect_wave_divergence(wave_df(lobes, side, **kw), side, rules)


class Indicators(unittest.TestCase):
    def test_ema200_column(self):
        df = wave_df(W)[["open", "high", "low", "close", "volume"]]
        self.assertIn("ema200", m.compute_indicators(df).columns)

    def test_rsi_extremes(self):
        up = pd.Series(np.arange(1.0, 60))
        self.assertEqual((m.rsi(up).iloc[-1], m.rsi(100 - up).iloc[-1]), (100.0, 0.0))


class Crosses(unittest.TestCase):
    def test_one_cross_per_lobe_boundary(self):
        self.assertEqual(crosses(wave_df(W)), CROSSES)

    def test_tiny_double_dot_does_not_split_a_trough(self):
        df = wave_df(W)
        df.iloc[15, df.columns.get_loc("macd")] = 0.01     # pokes above signal for one bar
        self.assertEqual(crosses(df), CROSSES)

    def test_wave_shorter_than_three_bars_is_merged(self):
        df = wave_df(W)
        df.iloc[15:17, df.columns.get_loc("macd")] = 1.0   # a decisive but 2-bar bull wave
        self.assertEqual(crosses(df), CROSSES)


class Detection(unittest.TestCase):
    def test_w_shape_is_found_on_its_own_side_only(self):
        for side, other in (("bullish", "bearish"), ("bearish", "bullish")):
            df = wave_df(W, side)
            out = m.detect_wave_divergence(df, side, RULES)
            self.assertEqual([(d["a_m"], d["b_m"], d["bump_bar"], d["signal_bar"]) for d in out],
                             [(15, 37, 26, 44)], side)
            self.assertEqual(out[0]["crosses"], crosses(df))
            self.assertEqual(m.detect_wave_divergence(df, other, RULES), [], side)

    def test_levels_come_from_wicks_and_the_macd_line(self):
        d = detect(W)[0]
        self.assertEqual((round(d["a_price"], 2), round(d["b_price"], 2)), (95.04, 92.07))
        self.assertEqual((round(d["a_macd"], 2), round(d["b_macd"], 2)), (-1.98, -0.99))
        d = detect(W, "bearish")[0]
        self.assertEqual((round(d["a_price"], 2), round(d["b_price"], 2)), (104.96, 107.93))
        self.assertEqual((round(d["a_macd"], 2), round(d["b_macd"], 2)), (1.98, 0.99))

    def test_not_reported_until_wave_b_is_closed_by_a_cross(self):
        df = wave_df(W)
        self.assertEqual(m.detect_wave_divergence(df.iloc[:44], "bullish", RULES), [])
        self.assertEqual(len(m.detect_wave_divergence(df.iloc[:45], "bullish", RULES)), 1)

    def test_macd_trough_must_be_higher(self):
        self.assertEqual(detect([(10, 1, 0), (12, 1.0, 5), (10, 1.2, 0), (12, 2.0, 8), (8, 1, 0)]), [])

    def test_price_low_must_be_lower(self):
        self.assertEqual(detect([(10, 1, 0), (12, 2.0, 8), (10, 1.2, 0), (12, 1.0, 5), (8, 1, 0)]), [])

    def test_a_bump_too_small_to_count_gives_nothing(self):
        self.assertEqual(detect([(10, 1, 0), (12, 6.0, 5), (10, 0.2, 0), (12, 1.0, 8), (8, 1, 0)]), [])

    def test_troughs_must_be_8_to_80_bars_apart(self):
        self.assertEqual(detect([(10, 1, 0), (12, 2.0, 5), (80, 1.2, 0), (12, 1.0, 8), (8, 1, 0)]), [])
        self.assertEqual(detect([(10, 1, 0), (4, 2.0, 5), (3, 1.2, 0), (4, 1.0, 8), (8, 1, 0)]), [])

    def test_strict_needs_both_troughs_below_zero(self):
        strict = dict(RULES, zero_line=True)
        self.assertEqual(len(detect(W, rules=strict)), 1)
        self.assertEqual(detect(W, rules=strict, offset=5.0), [])
        self.assertEqual([d["zero_ok"] for d in detect(W, offset=5.0)], [False])

    def test_triple_when_the_trough_before_a_also_diverges(self):
        lobes = [(10, 1, 0), (12, 3.0, 3), (10, 1.2, 0), (12, 2.0, 5), (10, 1.2, 0), (12, 1.0, 8), (8, 1, 0)]
        self.assertEqual([(d["a_m"], d["triple"]) for d in detect(lobes)], [(15, False), (37, True)])

    def test_structure_facts_for_the_score(self):
        d = detect(W)[0]
        self.assertEqual((round(d["bump_frac"], 2), round(d["macd_frac"], 2), round(d["price_atr"], 2)),
                         (0.69, 0.31, 2.97))
        self.assertEqual((d["wick"], d["vol_dry"]), (True, False))
        df = wave_df(W)
        df.iloc[10:22, df.columns.get_loc("volume")] = 3.0          # wave A traded 3x wave B
        self.assertTrue(m.detect_wave_divergence(df, "bullish", RULES)[0]["vol_dry"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `python -m unittest tests.test_waves -v`
Expected: errors, `module 'macd_divergence_screener_v5' has no attribute 'rsi'` / `'macd_crosses'` / `'detect_wave_divergence'`.

- [ ] **Step 4: Add the constants**

In `macd_divergence_screener_v5.py`, directly after the line `ATR_PERIOD  = 14`, add:

```python
EMA_TREND   = 200
RSI_PERIOD  = 14

# ── Wave detection (spec v2, Steps 1-4) ───────────────────────────────────────
MIN_WAVE    = 3        # bars; a shorter wave is merged into its neighbours
WHIP_FRAC   = 0.05     # a cross counts once |MACD - signal| reaches this share...
WHIP_RANGE  = 50       # ...of the MACD range over this many bars...
WHIP_BARS   = 3        # ...within this many bars, the cross bar being the first
SPACING     = (8, 80)  # bars allowed between MACD trough A and MACD trough B
BUMP_MIN    = 0.25     # height of the middle of the W, as a share of the leg's MACD range
PRICE_TAIL  = 2        # a wave's price extreme may print up to 2 bars after the wave ends
MAX_BUMPS   = 1        # 2 = the spec's extended mode (off by default)
```

- [ ] **Step 5: Add EMA 200 and RSI**

In `compute_indicators`, change the docstring's first line to `"""MACD line / signal / histogram, ATR14 (Wilder) and EMA 200, on completed bars."""` and replace its `return` statement with:

```python
    return pd.DataFrame({"macd": ml, "signal": sl, "hist": ml - sl, "atr": atr,
                         "ema200": ema(close, EMA_TREND)}, index=df.index)
```

Directly after `compute_indicators`, add:

```python
def rsi(close, p=RSI_PERIOD):
    """Wilder RSI. Used on BTC 4H only, for the regime tag."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / p, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)
```

- [ ] **Step 6: Add the detector**

Directly after the `swing_pivots` function, add:

```python
# ── Wave divergence (spec v2) ─────────────────────────────────────────────────
def macd_crosses(macd, signal):
    """Confirmed MACD/signal crossovers as [(bar, +1 bullish | -1 bearish)].

    The list alternates direction, so consecutive entries bound one wave. A
    cross that never opens a real gap (the whipsaw filter) is dropped, and a
    wave shorter than MIN_WAVE bars is removed together with the two crosses
    around it. Either way the waves on both sides join into one, which is what
    stops a tiny double dot from splitting one trough into two.
    """
    d = macd - signal
    m = pd.Series(macd)
    rng = (m.rolling(WHIP_RANGE, min_periods=1).max()
           - m.rolling(WHIP_RANGE, min_periods=1).min()).values
    raw = np.flatnonzero(((d[1:] > 0) & (d[:-1] <= 0)) | ((d[1:] < 0) & (d[:-1] >= 0))) + 1
    out = []
    for i in raw:
        i = int(i)
        s = 1 if d[i] > 0 else -1
        if out and out[-1][1] == s:
            continue                                   # already on that side
        if (s * d[i:i + WHIP_BARS]).max() < WHIP_FRAC * rng[i]:
            continue                                   # whipsaw
        if out and i - out[-1][0] < MIN_WAVE:
            out.pop()                                  # the wave it would close is too short
            continue
        out.append((i, s))
    return out


def detect_wave_divergence(df, side, preset):
    """REGULAR divergences of one side, read trough to trough on the MACD line.

    Bullish : two bear waves (MACD below signal) A then B with one bull wave,
              the bump, between them. Price low B < price low A while MACD
              trough B > MACD trough A.
    Bearish : the same picture upside down, so it is computed as the bullish
              case on a negated MACD line and negated highs.

    A divergence exists only once wave B has been closed by a cross; that
    cross is the signal bar. The function reads nothing but `df`, so calling
    it on df.iloc[:i + 1] is exactly what was knowable at bar i. The live scan
    and the backtest both rely on that.
    """
    s = 1 if side == "bullish" else -1
    macd = df["macd"].values * s
    ext = df["low"].values if s > 0 else -df["high"].values
    o, c, h, l = (df[k].values for k in ("open", "close", "high", "low"))
    vol, atr = df["volume"].values, df["atr"].values
    n = len(df)
    crosses = macd_crosses(df["macd"].values, df["signal"].values)

    waves = []                                         # closed waves on the trough side
    for (c0, d0), (c1, _) in zip(crosses, crosses[1:]):
        if d0 != -s:
            continue
        tm = c0 + int(np.argmin(macd[c0:c1]))
        tp = c0 + int(np.argmin(ext[c0:min(c1 + PRICE_TAIL, n)]))
        waves.append({"start": c0, "end": c1, "tm": tm, "tp": tp,
                      "m": float(macd[tm]), "p": float(ext[tp])})

    out = []
    for k in range(1, len(waves)):
        B = waves[k]
        for j in range(k - 1, max(k - 1 - MAX_BUMPS, -1), -1):
            A = waves[j]
            if not (B["p"] < A["p"] and B["m"] > A["m"]):
                continue
            if not (SPACING[0] <= B["tm"] - A["tm"] <= SPACING[1]):
                continue
            seg = macd[A["tm"]:B["tm"] + 1]
            leg = float(seg.max() - seg.min())
            if leg <= 0:
                continue
            bump = A["end"] + int(np.argmax(macd[A["end"]:B["start"]]))
            bump_frac = (float(macd[bump]) - B["m"]) / leg      # B is the higher trough
            if bump_frac < BUMP_MIN:
                continue
            tp = B["tp"]
            atr_pct = atr[tp] / c[tp] * 100 if c[tp] else 0.0
            if (A["p"] - B["p"]) / abs(A["p"]) * 100 < max(preset["min_price_pct"],
                                                         preset["atr_frac"] * atr_pct):
                continue
            macd_frac = (B["m"] - A["m"]) / leg
            if macd_frac < preset["min_macd_frac"]:
                continue
            zero_ok = B["m"] < 0                                # then A, the lower one, is too
            if preset["zero_line"] and not zero_ok:
                continue
            C = waves[j - 1] if j else None
            bar_rng = h[tp] - l[tp]
            wick = (min(o[tp], c[tp]) - l[tp]) if s > 0 else (h[tp] - max(o[tp], c[tp]))
            out.append({
                "side": side, "a_m": A["tm"], "b_m": B["tm"], "a_p": A["tp"], "b_p": tp,
                "a_macd": A["m"] * s, "b_macd": B["m"] * s,
                "a_price": abs(A["p"]), "b_price": abs(B["p"]),
                "a_start": A["start"], "b_start": B["start"], "signal_bar": B["end"],
                "bump_bar": bump, "bump_frac": bump_frac, "macd_frac": macd_frac,
                "price_atr": float((A["p"] - B["p"]) / atr[tp]) if atr[tp] > 0 else 0.0,
                "zero_ok": bool(zero_ok),
                "triple": bool(C and A["p"] < C["p"] and A["m"] > C["m"]),
                "vol_dry": bool(vol[B["start"]:B["end"]].mean() < vol[A["start"]:A["end"]].mean()),
                "wick": bool(bar_rng > 0 and wick >= 0.5 * bar_rng),
                "crosses": [x for x in crosses if A["start"] <= x[0] <= B["end"]],
            })
            break
    return out
```

- [ ] **Step 7: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass, including the untouched older tests (the old detector is still in place).

- [ ] **Step 8: Commit**

```bash
git add macd_divergence_screener_v5.py tests/helpers.py tests/test_waves.py
git commit -m "Add the MACD crossover-wave divergence detector"
```

---

### Task 2: Signal record, grade, regime and verdict

Swaps the scan over to the new detector and deletes the old one. After this task the app runs end to end on wave divergences, graded without outside context (no higher timeframe, OI, BTC or trigger yet; Task 4 adds those).

**Files:**
- Modify: `macd_divergence_screener_v5.py` (`PIVOT_SYNC`/`TF_CFG`/`STRICTNESS` ~L112-132; `MOVED_PCT` ~L193; delete `swing_pivots`, `_macd_anchor`, `detect_divergences`; replace `_mk`, `build_targets`, `setup_score`, `entry_verdict`; `build_rows`; `signals_for_tf`; one line in `scan_pair`)
- Rewrite: `tests/test_divergence.py`
- Modify: `tests/test_rows.py`

**Interfaces:**
- Consumes: `detect_wave_divergence`, `W`, `LONG`, `wave_df`.
- Produces:
  - `TF_CFG[tf]` keys: `label, minutes, bars, fresh, htf_minutes, ltf, ltf_minutes`.
  - `_mk(df, d, cfg, forming=False) -> dict` (signal record; private keys `_b` = signal bar index, `_d` = the detection dict).
  - `score_signal(sig, htf_ok=False, oi_chg=None, fund=None) -> (score:int, grade:"A"|"B"|"C")`.
  - `regime_tag(side, above_ema200, btc_rsi) -> "WITH TREND"|"COUNTER TREND"`.
  - `build_targets(sig) -> dict` with `entry_ref_price, targets_from ("signal"|"trigger"), risk_pct, targets {"1r","2r","3r","structure"}`.
  - `finish_signal(sig, htf_ok=False, oi_chg=None, fund=None, btc_rsi=None, trigger=None) -> sig`; `trigger` is `(price, trigger_bar_close_ms)` or `None`. Sets `score, grade, htf_ok, oi_chg_pct, regime, trigger_price, trigger_first, tags` and the `build_targets` keys. Safe to call twice.
  - `entry_verdict(sig, hours_to_event=None, event_title="") -> {"verdict": "enter"|"wait"|"skip", "reason", "level"}`.
  - `signals_for_tf(df, tf, rules, live_price, sym, wk_vol, forming_df=None) -> list[sig]`; each sig still carries `_df` (the frame it was detected on) and `_d`.
  - Row dicts from `build_rows` gain `grade` and `regime`; `score` is now the primary signal's score.

- [ ] **Step 1: Rewrite `tests/test_divergence.py`**

Replace the whole file with:

```python
"""Tests for pivots, config, the signal record, the score and the regime tag.

Run with:  python -m unittest tests.test_divergence -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import LONG, wave_df  # noqa: E402

RULES = m.STRICTNESS["balanced"]


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

    def test_signal_goes_stale_after_6_bars_on_4h_and_12_on_1h(self):
        self.assertEqual((m.TF_CFG["4h"]["fresh"], m.TF_CFG["1h"]["fresh"]), (6, 12))

    def test_the_pivot_detector_is_gone(self):
        for name in ("detect_divergences", "swing_pivots", "setup_score", "PIVOT_SYNC"):
            self.assertFalse(hasattr(m, name), name)


class SignalsForTf(unittest.TestCase):
    def run_tf(self, df, tf="1h", **kw):
        return m.signals_for_tf(df, tf, RULES, 100.5, "ENS/USDT:USDT", 9e7, **kw)

    def test_finished_signal_carries_everything_a_row_needs(self):
        out = self.run_tf(wave_df(LONG))            # 102 bars, signal bar 94
        self.assertEqual(len(out), 1)
        s = out[0]
        self.assertEqual((s["tf"], s["pair"], s["base"], s["side"], s["weekly_volume"]),
                         ("1h", "ENS/USDT:USDT", "ENS", "bullish", 9e7))
        self.assertEqual((s["bars_ago"], s["fresh"], s["forming"], s["status"]),
                         (7, True, False, "active"))
        self.assertEqual((s["signal_price"], s["stop_price"], s["pivot2_price"]),
                         (100.5, 91.8184, 92.0684))     # stop = low B - 0.25 ATR
        self.assertEqual((s["score"], s["grade"], s["regime"], s["tags"]),
                         (53, "C", "COUNTER TREND", ["W-PATTERN"]))
        self.assertEqual((s["targets_from"], s["targets"]["structure"]), ("signal", 101.0))
        for k in ("targets", "risk_pct", "current_price", "signal_time", "pivot1_time"):
            self.assertIn(k, s)
        self.assertNotIn("_b", s)

    def test_same_signal_is_stale_on_4h(self):
        self.assertEqual([s["fresh"] for s in self.run_tf(wave_df(LONG), "4h")], [False])

    def test_too_few_bars_gives_nothing(self):
        self.assertEqual(self.run_tf(wave_df(LONG).iloc[:60]), [])

    def test_signal_bar_still_forming_is_provisional(self):
        df = wave_df(LONG)
        closed, with_open = df.iloc[:94], df.iloc[:95]     # bar 94 is the closing cross
        self.assertEqual(self.run_tf(closed), [])
        out = self.run_tf(closed, forming_df=with_open)
        self.assertEqual([(s["forming"], s["bars_ago"]) for s in out], [(True, 0)])


FULL = dict(side="bullish", price_atr=2.0, macd_frac=0.5, bump_frac=0.75,
            zero_ok=True, vol_dry=True, wick=True, triple=True)
BARE = dict(FULL, zero_ok=False, vol_dry=False, wick=False, triple=False)


class Score(unittest.TestCase):
    def test_everything_true_is_100(self):
        self.assertEqual(m.score_signal(FULL, True, -3.0, 0.0), (100, "A"))

    def test_structure_alone_tops_out_at_40(self):
        self.assertEqual(m.score_signal(BARE), (40, "C"))

    def test_oi_rising_against_the_signal_costs_10(self):
        self.assertEqual(m.score_signal(FULL, True, 9.0, 0.02)[0], 80)      # 90 without OI
        self.assertEqual(m.score_signal(dict(FULL, side="bearish"), True, 9.0, -0.02)[0], 80)
        self.assertEqual(m.score_signal(FULL, True, 3.0, 0.02)[0], 90)      # under 5%: no penalty

    def test_bearish_oi_bonus_needs_funding_at_or_above_zero(self):
        bear = dict(FULL, side="bearish")
        self.assertEqual(m.score_signal(bear, True, -3.0, 0.01)[0], 100)
        self.assertEqual(m.score_signal(bear, True, -3.0, -0.01)[0], 90)

    def test_grade_cutoffs(self):
        self.assertEqual(m.score_signal(dict(BARE, zero_ok=True, wick=True), True)[1], "A")   # 75
        self.assertEqual(m.score_signal(BARE, True)[1], "B")                                  # 55
        self.assertEqual(m.score_signal(dict(BARE, zero_ok=True))[1], "C")                    # 50


class Regime(unittest.TestCase):
    def test_with_trend_needs_btc_and_the_coin_to_agree(self):
        self.assertEqual(m.regime_tag("bullish", True, 60.0), "WITH TREND")
        self.assertEqual(m.regime_tag("bullish", False, 60.0), "COUNTER TREND")
        self.assertEqual(m.regime_tag("bullish", True, 40.0), "COUNTER TREND")
        self.assertEqual(m.regime_tag("bearish", False, 40.0), "WITH TREND")
        self.assertEqual(m.regime_tag("bearish", True, 40.0), "COUNTER TREND")

    def test_unknown_btc_is_counter_trend(self):
        self.assertEqual(m.regime_tag("bullish", True, None), "COUNTER TREND")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Update `tests/test_rows.py`**

Replace the `SIG` constant and the whole `EntryVerdict` class with:

```python
SIG = {"status": "active", "fresh": True, "bars_ago": 2, "r_reached": 0.0,
       "pct_chg": 0.4, "signal_price": 100.0, "stop_price": 97.0, "risk_pct": 3.0,
       "grade": "A", "score": 74, "regime": "WITH TREND",
       "forming": False, "trigger_first": False}


def sig(**kw):
    return dict(SIG, **kw)


class EntryVerdict(unittest.TestCase):
    def verdict(self, s, *a):
        return m.entry_verdict(s, *a)["verdict"]

    def test_confirmed_yes_when_every_check_passes(self):
        v = m.entry_verdict(sig())
        self.assertEqual((v["verdict"], v["level"]), ("enter", 97.0))
        self.assertTrue(v["reason"].startswith("Yes (confirmed)."))
        self.assertIn("Risk 3.0%", v["reason"])

    def test_invalidated_is_no(self):
        v = m.entry_verdict(sig(status="invalidated"))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("hit the stop", v["reason"])

    def test_ran_is_no(self):
        v = m.entry_verdict(sig(status="ran", r_reached=2.5))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("2.5x the risk", v["reason"])

    def test_old_signal_bar_is_no(self):
        v = m.entry_verdict(sig(fresh=False, bars_ago=20))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("20 bars ago", v["reason"])

    def test_grade_c_is_no(self):
        v = m.entry_verdict(sig(grade="C", score=41))
        self.assertEqual(v["verdict"], "skip")
        self.assertIn("grade C", v["reason"])

    def test_counter_trend_needs_grade_a(self):
        v = m.entry_verdict(sig(regime="COUNTER TREND", grade="B"))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("Watch only", v["reason"])
        self.assertEqual(self.verdict(sig(regime="COUNTER TREND", grade="A")), "enter")

    def test_extended_waits_for_a_pullback_to_the_signal_price(self):
        v = m.entry_verdict(sig(status="extended", pct_chg=3.1))
        self.assertEqual((v["verdict"], v["level"]), ("wait", 100.0))
        self.assertIn("3.1%", v["reason"])

    def test_event_inside_24h_is_wait_and_names_the_event(self):
        v = m.entry_verdict(sig(), 14.2, "CPI m/m")
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("CPI m/m is due in 14h", v["reason"])

    def test_event_beyond_24h_or_unknown_does_not_block(self):
        self.assertEqual(self.verdict(sig(), 30.0, "CPI m/m"), "enter")
        self.assertEqual(self.verdict(sig(), None), "enter")

    def test_forming_bar_is_early_only_with_the_trigger(self):
        v = m.entry_verdict(sig(forming=True, trigger_first=True))
        self.assertEqual(v["verdict"], "enter")
        self.assertTrue(v["reason"].startswith("Yes (early)."))
        v = m.entry_verdict(sig(forming=True))
        self.assertEqual(v["verdict"], "wait")
        self.assertIn("still forming", v["reason"])

    def test_rules_apply_in_order(self):
        self.assertEqual(self.verdict(sig(status="invalidated"), 2.0, "FOMC"), "skip")
        self.assertEqual(self.verdict(sig(grade="C", regime="COUNTER TREND")), "skip")
        self.assertIn("Watch only", m.entry_verdict(
            sig(grade="B", regime="COUNTER TREND", status="extended"))["reason"])
```

In the `BuildRows` class make these three changes:

1. In `test_same_side_on_both_timeframes_is_one_row_4h_first`, the expected tuple becomes `(["4h", "1h"], 70, ["1h", "4h"], "4h")` (the row now shows the primary signal's score, not the maximum).
2. Replace `test_wait_beats_skip` with:

```python
    def test_wait_beats_skip(self):
        r = m.build_rows([full_sig("4h", "bullish", fresh=False),
                          full_sig("1h", "bullish", status="extended")])[0]
        self.assertEqual((r["primary_tf"], r["entry"]["verdict"]), ("1h", "wait"))
```

3. Add:

```python
    def test_row_takes_grade_and_regime_from_the_primary_signal(self):
        r = m.build_rows([full_sig("4h", "bullish", status="invalidated", grade="A"),
                          full_sig("1h", "bullish", grade="B", score=61)])[0]
        self.assertEqual((r["primary_tf"], r["grade"], r["score"], r["regime"]),
                         ("1h", "B", 61, "WITH TREND"))
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `python -m unittest tests.test_divergence tests.test_rows -v`
Expected: failures and errors (`no attribute 'score_signal'`, `KeyError: 'grade'`, old detector still present).

- [ ] **Step 4: Replace the config**

Replace the `PIVOT_SYNC` line, the comment block above `TF_CFG`, `TF_CFG` and `STRICTNESS` (from `PIVOT_SYNC  = 3` through the closing `}` of `STRICTNESS`) with:

```python
STOP_ATR       = 0.25   # the stop sits this many ATR beyond price extreme B
OI_AGAINST_PCT = 5.0    # OI rising more than this into B, against the signal, costs points
GRADE_A, GRADE_B = 70, 55

# Per-timeframe settings. `fresh` is how many bars a closed signal bar stays
# enterable. htf / ltf are the timeframes used for agreement and for the
# early-entry trigger.
TF_CFG = {
    "4h": {"label": "4H", "minutes": 240, "bars": 720, "fresh": 6,
           "htf_minutes": 1440, "ltf": "1h", "ltf_minutes": 60},
    "1h": {"label": "1H", "minutes": 60,  "bars": 500, "fresh": 12,
           "htf_minutes": 240, "ltf": "15m", "ltf_minutes": 15},
}

# Minimum divergence size. Strict also wants both troughs below zero
# (both peaks above zero for bearish).
STRICTNESS = {
    "balanced": {"min_price_pct": 0.25, "atr_frac": 0.20, "min_macd_frac": 0.15,
                 "zero_line": False},
    "strict":   {"min_price_pct": 0.50, "atr_frac": 0.55, "min_macd_frac": 0.22,
                 "zero_line": True},
}
```

Change the `MOVED_PCT` line's comment to `# beyond this the setup is "extended"`.

- [ ] **Step 5: Delete the pivot detector**

Delete these functions entirely: `swing_pivots`, `_macd_anchor`, `detect_divergences`. Keep `find_pivots` (the lower-timeframe trigger in Task 3 uses it). Change the section comment `# ── Divergence detection ──…` if it is left orphaned.

- [ ] **Step 6: Replace `_mk`**

Replace the whole `_mk` function with:

```python
def _mk(df, d, cfg, forming=False):
    """Signal record for one detect_wave_divergence() hit `d` on frame `df`.

    The pivot1_* / pivot2_* names are kept from v5: 1 is A, 2 is B.
    """
    side, bull = d["side"], d["side"] == "bullish"
    n, sb, bp = len(df), d["signal_bar"], d["b_p"]
    sp = float(df["close"].iloc[sb])                 # signal price = close of the signal bar
    atrb = float(df["atr"].iloc[bp])
    if not atrb > 0:
        atrb = sp * 0.01
    stop = px(d["b_price"] - STOP_ATR * atrb) if bull else px(d["b_price"] + STOP_ATR * atrb)
    macd = df["macd"].values
    after = macd[sb + 1:]
    zero_cross = bool(len(after) and ((macd[sb] <= 0 and after.max() > 0) if bull
                                      else (macd[sb] >= 0 and after.min() < 0)))
    off = pd.Timedelta(hours=DISPLAY_TZ_OFFSET_H)

    def fmt(i):
        return (df.index[i] + off).strftime("%b %d %Y, %I:%M %p")

    def ms(i):
        return int(df.index[i].timestamp() * 1000)

    return {
        "side": side,
        "pivot1_time": fmt(d["a_p"]), "pivot2_time": fmt(bp), "signal_time": fmt(sb),
        "pivot1_price": px(d["a_price"]), "pivot2_price": px(d["b_price"]),
        "pivot1_macd": round(d["a_macd"], 8), "pivot2_macd": round(d["b_macd"], 8),
        "signal_price": px(sp), "stop_price": stop,
        "signal_risk": max((sp - stop) if bull else (stop - sp), sp * 0.005),
        "structure_target": px(float(df["high" if bull else "low"].iloc[d["bump_bar"]])),
        "bars_ago": n - 1 - sb, "fresh": (n - 1 - sb) <= cfg["fresh"], "forming": forming,
        "b_start_ms": ms(d["b_start"]), "b_ms": ms(bp), "signal_ms": ms(sb),
        "signal_close_ms": ms(sb) + cfg["minutes"] * 60_000,
        "above_ema200": bool(sp > float(df["ema200"].iloc[sb])),
        "price_atr": round(d["price_atr"], 3), "macd_frac": round(d["macd_frac"], 4),
        "bump_frac": round(d["bump_frac"], 4),
        "zero_ok": d["zero_ok"], "triple": d["triple"], "vol_dry": d["vol_dry"],
        "wick": d["wick"], "zero_cross": zero_cross,
        "_b": sb, "_d": d,
    }
```

Do not change `evaluate_status`. `_b` is now the signal bar, so its "bars after" already means "after the signal bar".

- [ ] **Step 7: Replace `build_targets` and `setup_score`**

Replace the whole `build_targets` function and the whole `setup_score` function (and its `# ── Setup score ──` comment) with:

```python
def build_targets(sig):
    """1R / 2R / 3R and the structure target, from the entry reference.

    The entry reference is the signal price, or the lower-timeframe trigger
    price when the trigger fired before the signal bar closed.
    """
    k = 1 if sig["side"] == "bullish" else -1
    stop = sig["stop_price"]
    early = bool(sig.get("trigger_first"))
    ref = sig["trigger_price"] if early else sig["signal_price"]
    risk = max((ref - stop) * k, ref * 0.005)
    return {"entry_ref_price": px(ref),
            "targets_from": "trigger" if early else "signal",
            "risk_pct": round(risk / ref * 100, 2),
            "targets": {"1r": px(ref + k * risk), "2r": px(ref + 2 * k * risk),
                        "3r": px(ref + 3 * k * risk),
                        "structure": sig["structure_target"]}}


# ── Quality score, regime, tags ────────────────────────────────────────────────
def score_signal(sig, htf_ok=False, oi_chg=None, fund=None):
    """(0-100 score, grade). A description of the setup, not a win rate.

    oi_chg is the % change in open interest into B, fund the funding rate
    (only its sign is used). Either one unknown: no OI points, no penalty.
    """
    s = min(sig["price_atr"], 1.0) * 15                                   # 0-15
    s += min(sig["macd_frac"] / 0.5, 1.0) * 15                            # 0-15
    s += float(np.clip((sig["bump_frac"] - BUMP_MIN) / 0.5, 0, 1)) * 10   # 0-10
    s += 10 * sig["zero_ok"] + 15 * bool(htf_ok) + 10 * sig["vol_dry"] + 10 * sig["wick"]
    s += 5 * sig["triple"]
    if oi_chg is not None and fund is not None:
        f = fund if sig["side"] == "bullish" else -fund
        if oi_chg < 0 and f <= 0:
            s += 10
        elif oi_chg > OI_AGAINST_PCT and f > 0:
            s -= 10
    score = int(max(0, min(100, round(s))))
    return score, ("A" if score >= GRADE_A else "B" if score >= GRADE_B else "C")


def regime_tag(side, above_ema200, btc_rsi):
    """WITH TREND when BTC 4H RSI and the coin's 200 EMA both agree with the side."""
    if btc_rsi is None:
        return "COUNTER TREND"
    ok = (btc_rsi > 50 and above_ema200) if side == "bullish" \
        else (btc_rsi < 50 and not above_ema200)
    return "WITH TREND" if ok else "COUNTER TREND"


def finish_signal(sig, htf_ok=False, oi_chg=None, fund=None, btc_rsi=None, trigger=None):
    """Score, grade, regime, trigger, tags and targets.

    Called once without context so a signal is always complete, and again by
    add_context() once the scan has fetched the outside data. `trigger` is
    (price, close time in ms of the trigger bar) or None.
    """
    sig["score"], sig["grade"] = score_signal(sig, htf_ok, oi_chg, fund)
    sig["htf_ok"] = bool(htf_ok)
    sig["oi_chg_pct"] = None if oi_chg is None else round(oi_chg, 1)
    sig["regime"] = regime_tag(sig["side"], sig["above_ema200"], btc_rsi)
    sig["trigger_price"] = px(trigger[0]) if trigger else None
    # "first" = the trigger bar closed before the signal bar did
    sig["trigger_first"] = bool(trigger and (sig["forming"] or trigger[1] < sig["signal_close_ms"]))
    sig["tags"] = [t for t, on in (("TRIPLE", sig["triple"]),
                                   ("W-PATTERN", sig["bump_frac"] >= 0.5),
                                   ("LTF TRIGGER", bool(trigger)),
                                   ("ZERO-LINE CROSS", sig["zero_cross"])) if on]
    sig.update(build_targets(sig))
    return sig
```

- [ ] **Step 8: Replace `entry_verdict`**

Replace the whole `entry_verdict` function with:

```python
def entry_verdict(sig, hours_to_event=None, event_title=""):
    """Is this setup enterable as of the scan? A CHECKLIST, not a prediction.

    First matching rule wins (spec "Good to enter?"):
      skip   stop hit · already ran 1R+ · signal bar too old · grade C
      wait   counter trend below grade A · price ran >MOVED_PCT from the signal ·
             a market-mover print inside EVENT_WAIT_H · signal bar still
             forming with no lower-timeframe trigger
      enter  none of the above: "Yes (early)" on a forming signal bar,
             "Yes (confirmed)" on a closed one

    `level` is a price the dashboard prints after the reason, so it is
    formatted in one place (sub-cent prices need plain digits).
    """
    def out(verdict, reason, level=None):
        return {"verdict": verdict, "reason": reason, "level": level}

    st = sig["status"]
    if st == "invalidated":
        return out("skip", "No. Price already hit the stop, so this divergence failed.")
    if st == "ran":
        return out("skip", f"No. Price already moved {sig['r_reached']:g}x the risk toward the "
                           "target. The move has happened; entering now is chasing it.")
    if not sig["fresh"]:
        return out("skip", f"No. The signal bar closed {sig['bars_ago']} bars ago, "
                           "too long ago to act on.")
    if sig["grade"] == "C":
        return out("skip", f"No. Weak setup: grade C, score {sig['score']} of 100.")
    if sig["regime"] == "COUNTER TREND" and sig["grade"] != "A":
        return out("wait", "Not yet. Watch only: this is against the trend, and a counter-trend "
                           f"setup needs grade A. This one is grade {sig['grade']}.")
    if st == "extended":
        return out("wait", f"Not yet. Price has already moved {sig['pct_chg']:g}% away from the "
                           "signal. Wait for it to pull back to",
                   sig["signal_price"])
    if hours_to_event is not None and hours_to_event <= EVENT_WAIT_H:
        return out("wait", f"Not yet. {event_title} is due in {hours_to_event:.0f}h and can "
                           "whip price. Wait until it has passed.")
    if sig["forming"] and not sig["trigger_first"]:
        return out("wait", "Not yet. The signal bar is still forming and the lower timeframe "
                           "has not triggered. Wait for the bar to close.")
    if sig["forming"]:
        return out("enter", "Yes (early). The lower timeframe triggered while the signal bar is "
                            f"still forming. Risk {sig['risk_pct']}% to the stop at",
                   sig["stop_price"])
    return out("enter", f"Yes (confirmed). The signal bar has closed, the grade is {sig['grade']} "
                        f"and the stop is untouched. Risk {sig['risk_pct']}% to the stop at",
               sig["stop_price"])
```

- [ ] **Step 9: Update `build_rows`**

In `build_rows`, replace the loop body that starts at `verdicts = {…}` (the second `for r in rows.values():` block) with:

```python
    for r in rows.values():
        verdicts = {tf: entry_verdict(r["signals"][tf], hours_to_event, event_title)
                    for tf in r["tfs"]}
        best = min(r["tfs"], key=lambda tf: _VERDICT_RANK[verdicts[tf]["verdict"]])
        p = r["signals"][best]
        r["primary_tf"] = best
        r["score"], r["grade"], r["regime"] = p["score"], p["grade"], p["regime"]
        r["fresh"]  = any(s["fresh"] for s in r["signals"].values())
        r["status"] = p["status"]
        r["entry"]  = verdicts[best]
    return list(rows.values())
```

- [ ] **Step 10: Replace `signals_for_tf`**

Replace the whole `signals_for_tf` function with:

```python
def signals_for_tf(df, tf, rules, live_price, sym, wk_vol, forming_df=None):
    """Finished signals for one pair on one timeframe.

    `df` holds completed candles with the indicator columns. `forming_df` is
    the same frame plus the still-open candle (or None); it contributes only a
    signal whose signal bar is that open candle, flagged `forming`. Fewer than
    80 bars is not enough for MACD to have settled.

    Each signal keeps two private keys for the caller: `_df` (the frame it was
    found on) and `_d` (the raw detection), used for the chart.
    """
    if df is None or len(df) < 80:
        return []
    cfg, found = TF_CFG[tf], []
    for src, forming in ((forming_df, True), (df, False)):
        if src is None:
            continue
        for side in ("bullish", "bearish"):
            for d in detect_wave_divergence(src, side, rules):
                if forming and d["signal_bar"] != len(src) - 1:
                    continue
                sig = _mk(src, d, cfg, forming)
                sig.update(evaluate_status(src, sig, live_price))
                sig["_df"] = src
                found.append(sig)
    out = []
    for sig in dedupe(found):
        sig.pop("_b")
        sig.update({
            "pair": sym, "base": sym.split("/")[0], "tf": tf,
            "current_price": px(live_price),
            "weekly_volume": round(wk_vol, 2) if wk_vol else None,
        })
        out.append(finish_signal(sig))
    return out
```

In `dedupe`'s docstring, change "Successive pivots" to "Successive divergences". The code is unchanged.

- [ ] **Step 11: Keep the scan serialisable**

In `scan_route` → `scan_pair`, directly after

```python
                if not sigs:
                    return
```

add (Task 4 replaces this line with the chart cache):

```python
                for s in sigs:                      # frames cannot go over the wire
                    s.pop("_df"); s.pop("_d")
```

- [ ] **Step 12: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 13: Smoke-test a real scan**

Run: `python macd_divergence_screener_v5.py`, open `http://localhost:5099`, type `BTC, ETH, SOL, ENS` into *Specific coins*, press *Run Scan*.
Expected: the scan finishes without a "skipped:" line in Activity. Rows may look odd (the table is rebuilt in Task 6) but each row's answer starts with "Yes", "Not yet" or "No". Stop the server.

- [ ] **Step 14: Commit**

```bash
git add macd_divergence_screener_v5.py tests/test_divergence.py tests/test_rows.py
git commit -m "Switch the scan to wave divergences with grade, regime and the v2 checklist"
```

---

### Task 3: Context helpers (higher timeframe, trigger, OI)

**Files:**
- Modify: `macd_divergence_screener_v5.py` (new functions directly after `finish_signal`)
- Create: `tests/test_context.py`

**Interfaces:**
- Consumes: `finish_signal`, `find_pivots`, `TF_CFG`, `signals_for_tf`.
- Produces:
  - `asof(series, ts) -> float | None` (last value at or before `ts`).
  - `htf_agrees(htf, close_ts, htf_minutes, side) -> bool`; `htf` is a frame with a `macd` column indexed by bar open time, or `None`.
  - `ltf_trigger(ltf, side, since_ts, from_ts, ltf_minutes) -> (price, close_ms) | None`; `ltf` needs `high low close macd signal`.
  - `oi_change_pct(hist, t0_ms, t1_ms) -> float | None`; `hist` is ccxt OI history (dicts with `timestamp` and `openInterestAmount` and/or `openInterestValue`).
  - `add_context(sig, htf, ltf, oi_hist, fund, btc) -> sig`; `btc` is a frame with an `rsi` column indexed by BTC 4H bar open time, or `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context.py`:

```python
"""Tests for the outside context a signal is graded with.

Run with:  python -m unittest tests.test_context -v
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import macd_divergence_screener_v5 as m  # noqa: E402
from tests.helpers import LONG, wave_df  # noqa: E402

T = pd.Timestamp


class HigherTimeframe(unittest.TestCase):
    D = pd.DataFrame({"macd": [1.0, 2.0, 1.5]}, index=pd.date_range("2026-01-01", periods=3, freq="D"))

    def test_uses_only_bars_completed_by_the_signal_close(self):
        # at Jan 3 00:00 the Jan 1 and Jan 2 bars are complete: 1.0 -> 2.0 rising
        self.assertTrue(m.htf_agrees(self.D, T("2026-01-03"), 1440, "bullish"))
        # a day later the Jan 3 bar counts too: 2.0 -> 1.5 falling
        self.assertFalse(m.htf_agrees(self.D, T("2026-01-04"), 1440, "bullish"))
        self.assertTrue(m.htf_agrees(self.D, T("2026-01-04"), 1440, "bearish"))

    def test_too_little_history_or_none_is_no_agreement(self):
        self.assertFalse(m.htf_agrees(self.D, T("2026-01-02"), 1440, "bullish"))
        self.assertFalse(m.htf_agrees(None, T("2026-01-04"), 1440, "bullish"))

    def test_asof(self):
        s = self.D["macd"]
        self.assertEqual((m.asof(s, T("2026-01-02 12:00")), m.asof(s, T("2025-12-31"))), (2.0, None))


def ltf_frame():
    """40 bars of 15m: swing high 105 at bar 10, MACD crosses up at bar 20,
    first close above the swing high at bar 24."""
    n = 40
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    high = np.full(n, 101.0); high[10] = 105; high[24:] = 107
    close = np.full(n, 100.0); close[24:] = 106
    d = np.full(n, -1.0); d[20:] = 1
    return pd.DataFrame({"high": high, "low": 99.0, "close": close, "macd": d, "signal": 0.0},
                        index=idx)


class Trigger(unittest.TestCase):
    def test_fires_on_the_close_above_the_swing_high_after_the_cross(self):
        f = ltf_frame()
        close_ms = int(f.index[25].timestamp() * 1000)          # bar 24 closes when bar 25 opens
        self.assertEqual(m.ltf_trigger(f, "bullish", f.index[5], f.index[18], 15), (106.0, close_ms))

    def test_a_cross_before_the_signal_bar_does_not_count(self):
        f = ltf_frame()
        self.assertIsNone(m.ltf_trigger(f, "bullish", f.index[5], f.index[22], 15))

    def test_the_swing_high_must_be_after_price_low_b(self):
        f = ltf_frame()
        self.assertIsNone(m.ltf_trigger(f, "bullish", f.index[12], f.index[18], 15))

    def test_bearish_is_the_mirror(self):
        f = ltf_frame()
        mir = pd.DataFrame({"high": 201.0, "low": 200 - f["high"], "close": 200 - f["close"],
                            "macd": -f["macd"], "signal": 0.0}, index=f.index)
        self.assertEqual(m.ltf_trigger(mir, "bearish", f.index[5], f.index[18], 15)[0], 94.0)


class OpenInterest(unittest.TestCase):
    HIST = [{"timestamp": 1000, "openInterestAmount": 100},
            {"timestamp": 2000, "openInterestAmount": 90}]

    def test_change_between_two_times(self):
        self.assertAlmostEqual(m.oi_change_pct(self.HIST, 1500, 2500), -10.0)

    def test_none_when_history_does_not_cover_both_times(self):
        self.assertIsNone(m.oi_change_pct(self.HIST, 500, 2500))      # nothing at or before t0
        self.assertIsNone(m.oi_change_pct(self.HIST, 1500, 1800))     # same point for both
        self.assertIsNone(m.oi_change_pct(None, 1500, 2500))

    def test_falls_back_to_usd_value(self):
        hist = [{"timestamp": 1000, "openInterestValue": 50.0},
                {"timestamp": 2000, "openInterestValue": 60.0}]
        self.assertAlmostEqual(m.oi_change_pct(hist, 1000, 2000), 20.0)


class AddContext(unittest.TestCase):
    def setUp(self):
        self.sig = m.signals_for_tf(wave_df(LONG), "1h", m.STRICTNESS["balanced"],
                                    100.5, "ENS/USDT:USDT", 9e7)[0]
        idx = pd.date_range("2025-12-25", periods=80, freq="4h")
        self.h4 = pd.DataFrame({"macd": np.arange(80.0)}, index=idx)        # rising
        self.btc = pd.DataFrame({"rsi": 60.0}, index=idx)

    def test_without_context_it_is_the_bare_structure_grade(self):
        self.assertEqual((self.sig["score"], self.sig["grade"], self.sig["regime"]),
                         (53, "C", "COUNTER TREND"))

    def test_context_lifts_the_grade_and_sets_the_regime(self):
        s = self.sig
        hist = [{"timestamp": s["b_start_ms"] - 1, "openInterestAmount": 100},
                {"timestamp": s["b_ms"] - 1, "openInterestAmount": 90}]
        m.add_context(s, self.h4, None, hist, 0.0, self.btc)
        self.assertEqual((s["score"], s["grade"], s["regime"], s["htf_ok"], s["oi_chg_pct"]),
                         (78, "A", "WITH TREND", True, -10.0))
        self.assertEqual((s["tags"], s["trigger_price"], s["targets_from"]),
                         (["W-PATTERN"], None, "signal"))

    def test_missing_context_changes_nothing(self):
        m.add_context(self.sig, None, None, None, None, None)
        self.assertEqual((self.sig["score"], self.sig["regime"]), (53, "COUNTER TREND"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m unittest tests.test_context -v`
Expected: errors, `no attribute 'htf_agrees'` and friends. `test_without_context…` passes.

- [ ] **Step 3: Add the helpers**

Directly after `finish_signal`, add:

```python
# ── Outside context for a signal ───────────────────────────────────────────────
def asof(series, ts):
    """Last value at or before ts, or None."""
    s = series.loc[:ts]
    return float(s.iloc[-1]) if len(s) else None


def htf_agrees(htf, close_ts, htf_minutes, side):
    """Was the higher-timeframe MACD line rising (falling for bearish) on its
    last bar completed by close_ts? Frames are indexed by bar OPEN time, so a
    bar is complete once open + htf_minutes <= close_ts."""
    if htf is None:
        return False
    m = htf["macd"].loc[:close_ts - pd.Timedelta(minutes=htf_minutes)]
    if len(m) < 2:
        return False
    return bool(m.iloc[-1] > m.iloc[-2]) if side == "bullish" else bool(m.iloc[-1] < m.iloc[-2])


def ltf_trigger(ltf, side, since_ts, from_ts, ltf_minutes):
    """Early-entry trigger on the lower timeframe (spec Step 8).

    From from_ts (the signal bar's open) onward: the MACD has crossed in the
    signal's direction AND a candle closes beyond the last 3/3 swing high
    (low for bearish) made since since_ts (price extreme B). Returns
    (close of that candle, its close time in ms), or None.
    """
    k = 1 if side == "bullish" else -1
    d = ((ltf["macd"] - ltf["signal"]) * k).values
    close = ltf["close"].values
    src = ltf["high"].values if k > 0 else ltf["low"].values
    highs, lows = find_pivots(src, 3, 3)
    piv = [p for p in (highs if k > 0 else lows) if ltf.index[p] >= since_ts]
    crossed = False
    for i in range(max(1, int(ltf.index.searchsorted(from_ts))), len(ltf)):
        crossed = crossed or (d[i] > 0 and d[i - 1] <= 0)
        known = [p for p in piv if p + 3 < i]          # confirmed by bars before this one
        if crossed and known and k * close[i] > k * src[known[-1]]:
            return float(close[i]), int(ltf.index[i].timestamp() * 1000) + ltf_minutes * 60_000
    return None


def oi_change_pct(hist, t0_ms, t1_ms):
    """% change in open interest between two times, or None when the history
    does not cover both. One unit is used for both points (coin amount when the
    venue reports it, else USD value) so a price move is not read as OI."""
    # shortcut: nearest point at or before each time, so daily OI on a wave
    # shorter than a day gives None; fetch finer OI history if 4H scores need it.
    pts = sorted((p for p in (hist or []) if p.get("timestamp")), key=lambda p: p["timestamp"])
    for key in ("openInterestAmount", "openInterestValue"):
        a = [p for p in pts if p["timestamp"] <= t0_ms and p.get(key)]
        b = [p for p in pts if p["timestamp"] <= t1_ms and p.get(key)]
        if a and b and b[-1] is not a[-1]:
            return (float(b[-1][key]) / float(a[-1][key]) - 1) * 100.0
    return None


def add_context(sig, htf, ltf, oi_hist, fund, btc):
    """Re-grade a signal with everything fetched for it. Any argument may be
    None; that component then scores nothing."""
    cfg = TF_CFG[sig["tf"]]
    close_ts = pd.Timestamp(sig["signal_close_ms"], unit="ms")
    trig = None
    if ltf is not None and sig["fresh"]:
        trig = ltf_trigger(ltf, sig["side"], pd.Timestamp(sig["b_ms"], unit="ms"),
                           pd.Timestamp(sig["signal_ms"], unit="ms"), cfg["ltf_minutes"])
    btc_rsi = asof(btc["rsi"], close_ts - pd.Timedelta(minutes=240)) if btc is not None else None
    return finish_signal(sig, htf_agrees(htf, close_ts, cfg["htf_minutes"], sig["side"]),
                         oi_change_pct(oi_hist, sig["b_start_ms"], sig["b_ms"]),
                         fund, btc_rsi, trig)
```

- [ ] **Step 4: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add macd_divergence_screener_v5.py tests/test_context.py
git commit -m "Add higher-timeframe, trigger and OI context for grading a signal"
```

---

### Task 4: Wire the context into the scan

**Files:**
- Modify: `macd_divergence_screener_v5.py` (`_state` block ~L312; `fetch_oi_history`; `pair_context`; `load_state`; `universe_route`; `scan_route`)
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `add_context`, `signals_for_tf(..., forming_df)`, `rsi`, `compute_indicators`.
- Produces:
  - `HTF_BARS_1D = 200`, `LTF_BARS_15M = 300`, `STATE_VERSION = 2`.
  - `_charts: dict[str, (df, d, sig)]` keyed `"<pair>|<side>|<tf>"`, refilled by every scan.
  - `usdt_perps(mkts) -> list[str]`.
  - `with_indicators(df) -> df` and `forming_df(raw, closed) -> df | None`, now module level.
  - `fetch_btc_rsi(ex, limiter) -> DataFrame(rsi) | None`.
  - `fetch_oi_history(ex, limiter, sym, limit, tf="1d", tries=3)`.
  - `pair_context(...) -> (mover, flow, error, d1, oi_hist)`.
  - Scan meta gains `"v": 2`; `load_state` ignores any state without it.

- [ ] **Step 1: Write the failing tests**

In `tests/test_app.py`:

1. In `Routes.test_signals_returns_rows` nothing changes.
2. Add to class `Routes`:

```python
    def test_usdt_perps_keeps_only_active_crypto_linear_perps(self):
        mkts = {"ENS/USDT:USDT":  {"active": True,  "base": "ENS"},
                "USDC/USDT:USDT": {"active": True,  "base": "USDC"},
                "TSLA/USDT:USDT": {"active": True,  "base": "TSLA"},
                "OLD/USDT:USDT":  {"active": False, "base": "OLD"},
                "BTC/USD:BTC":    {"active": True,  "base": "BTC"}}
        self.assertEqual(m.usdt_perps(mkts), ["ENS/USDT:USDT"])
```

3. Replace the two tests in `SavedState` with:

```python
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
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m unittest tests.test_app -v`
Expected: `no attribute 'usdt_perps'`; `test_state_from_before_wave_detection_is_ignored` fails (the old rows are loaded).

- [ ] **Step 3: Constants, chart cache and small helpers**

After the `MOVERS_STATE_FILE` assignment add:

```python
STATE_VERSION = 2     # scan_state.json written by an older row shape is ignored
HTF_BARS_1D   = 200   # daily history, for the 4H signal's higher-timeframe check
LTF_BARS_15M  = 300   # 15m history, for the 1H signal's early-entry trigger
```

After the `_cal_state = {...}` assignment add:

```python
# Candles and geometry of the last scan's signals, for /chart.
# shortcut: memory only, so charts need a fresh scan after a restart; persist
# a slice per signal in scan_state.json if that becomes annoying.
_charts = {}
```

Directly after `drop_forming_candle` add:

```python
def with_indicators(df):
    return pd.concat([df, compute_indicators(df)], axis=1).dropna()


def forming_df(raw, closed):
    """The completed candles plus the still-open one, with indicators, or None
    when no candle is open. Only ever used to spot a signal bar that is still
    forming; everything else is computed on completed candles."""
    return with_indicators(raw) if len(raw) == len(closed) + 1 else None


def usdt_perps(mkts):
    """Active linear USDT perps, minus stablecoin bases and TradFi perps."""
    return [s for s in mkts
            if s.endswith("/USDT:USDT") and mkts[s].get("active")
            and s.split("/")[0] not in STABLE_BASES and not is_tradfi(mkts[s])]


def fetch_btc_rsi(ex, limiter):
    """BTC 4H RSI 14 on completed bars, for the regime tag. None when unavailable."""
    df = drop_forming_candle(
        fetch_df_deep(ex, limiter, "BTC/USDT:USDT", "4h", TF_CFG["4h"]["bars"]), 240)
    if df is None or len(df) < 80:
        return None
    return pd.DataFrame({"rsi": rsi(df["close"])}).dropna()
```

- [ ] **Step 4: `fetch_oi_history` takes a timeframe**

Change its signature and the one call inside it:

```python
def fetch_oi_history(ex, limiter, sym, limit, tf="1d", tries=3):
    """OI history with the same backoff discipline as fetch_ohlcv."""
```
```python
            return ex.fetch_open_interest_history(sym, timeframe=tf, limit=limit)
```

- [ ] **Step 5: `pair_context` returns the daily candles and the OI history**

Replace the whole function with:

```python
def pair_context(ex, limiter, sym, live_price, fund_day, wk_vol):
    """Movers + OI-flow context for a pair that HAS a divergence.

    Returns (mover, flow, error, daily candles, daily OI history). The last two
    are reused to grade the signal. A failure here must never cost the row
    itself, so it comes back as text.
    """
    try:
        df = drop_forming_candle(
            _to_df(fetch_ohlcv(ex, limiter, sym, "1d", limit=HTF_BARS_1D)), 1440)
        if df is None or df.empty:
            return None, None, "no daily candles", None, None
        mover = analyse_mover(df, live_price)
        if mover is not None:
            mover["flags"] = classify_mover(mover)[1]
        hist = fetch_oi_history(ex, limiter, sym, OI_LOOKBACK_D + 4)
        return mover, flow_context(hist, df, live_price, fund_day, wk_vol), None, df, hist
    except Exception as e:
        return None, None, str(e), None, None
```

- [ ] **Step 6: State version**

In `load_state`, replace

```python
        if "tfs" not in meta:
            # Written before rows existed (flat signal list, single timeframe).
            # The page cannot draw it, so start empty rather than broken.
```
with
```python
        if meta.get("v") != STATE_VERSION:
            # Written with an older row shape. The page cannot draw it, so
            # start empty rather than broken.
```

In `scan_route` → `persist()`, add `"v": STATE_VERSION,` as the first key of the `meta = {...}` dict.

- [ ] **Step 7: Use `usdt_perps` in both routes**

In `universe_route`, replace

```python
    for sym in mkts:
        if not (sym.endswith("/USDT:USDT") and mkts[sym].get("active")):
            continue
        base = sym.split("/")[0]
        if base in STABLE_BASES or is_tradfi(mkts[sym]):
            continue
        v = usd_volume(tickers.get(sym))
```
with
```python
    for sym in usdt_perps(mkts):
        base = sym.split("/")[0]
        v = usd_volume(tickers.get(sym))
```

In `scan_route`, replace the `perps = [s for s in mkts if … ]` list comprehension (4 lines) with `perps = usdt_perps(mkts)`.

- [ ] **Step 8: Wire the scan**

All inside `scan_route` → `generate()`:

(a) Directly after `limiter = RateLimiter(spd["rps"].get(exid, 8))` add:

```python
        btc = fetch_btc_rsi(main_ex, limiter)
        if btc is None:
            yield sse("log", {"msg": "BTC 4H candles unavailable: every setup is tagged COUNTER TREND"})
        _charts.clear()
```

(b) Delete the nested `def with_indicators(df): …` (it is module level now).

(c) In `scan_pair`, replace

```python
                df4 = with_indicators(df4)
```
with
```python
                f4 = forming_df(raw, df4)
                df4 = with_indicators(df4)
```

(d) Replace the 1H block

```python
                df1 = None
                raw1 = fetch_df_deep(ex, limiter, sym, "1h", TF_CFG["1h"]["bars"])
                if raw1 is not None and len(raw1) >= 80:
                    live_price = float(raw1["close"].iloc[-1])    # the newer tick
                    df1 = drop_forming_candle(raw1, TF_CFG["1h"]["minutes"])
                    df1 = with_indicators(df1) if df1 is not None else None
```
with
```python
                df1 = f1 = None
                raw1 = fetch_df_deep(ex, limiter, sym, "1h", TF_CFG["1h"]["bars"])
                if raw1 is not None and len(raw1) >= 80:
                    live_price = float(raw1["close"].iloc[-1])    # the newer tick
                    df1 = drop_forming_candle(raw1, TF_CFG["1h"]["minutes"])
                    if df1 is not None:
                        f1 = forming_df(raw1, df1)
                        df1 = with_indicators(df1)
```

(e) Replace everything from `sigs = (signals_for_tf(` down to and including the `for row in build_rows(…): q.put(("result", row))` loop with:

```python
                sigs = (signals_for_tf(df4, "4h", rules, live_price, sym, wk_vol, f4) +
                        signals_for_tf(df1, "1h", rules, live_price, sym, wk_vol, f1))
                if not sigs:
                    return
                # Everything below is fetched only for pairs worth a row.
                mover, flow, cerr, d1, oi_d = pair_context(ex, limiter, sym, live_price,
                                                           funding.get(sym), wk_vol)
                d1 = with_indicators(d1) if d1 is not None else None
                oi_h = df15 = None
                if any(s["tf"] == "1h" for s in sigs):
                    oi_h = fetch_oi_history(ex, limiter, sym, 100, tf="1h")
                if any(s["tf"] == "1h" and s["fresh"] for s in sigs):
                    df15 = drop_forming_candle(
                        fetch_df_deep(ex, limiter, sym, "15m", LTF_BARS_15M), 15)
                    df15 = with_indicators(df15) if df15 is not None and len(df15) >= 80 else None
                for s in sigs:
                    four = s["tf"] == "4h"
                    add_context(s, d1 if four else df4, df1 if four else df15,
                                oi_d if four else oi_h, funding.get(sym), btc)
                    _charts[f"{sym}|{s['side']}|{s['tf']}"] = (s.pop("_df"), s.pop("_d"), dict(s))
                for row in build_rows(sigs, mover, flow, cerr, ev_h, ev_title):
                    q.put(("result", row))
```

(This removes the temporary `s.pop("_df"); s.pop("_d")` loop from Task 2.)

- [ ] **Step 9: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 10: Smoke-test a real scan**

Run the app, scan `BTC, ETH, SOL, ENS, DOGE, XRP, SUI, LINK` on Binance.
Expected: no "skipped:" lines in Activity; `http://localhost:5099/signals` shows rows whose signals have `grade`, `regime` (`WITH TREND` appears on at least some when BTC RSI and the coin agree), `htf_ok`, `oi_chg_pct` (a number on at least one row, `null` is allowed), `tags`. Stop the server.

- [ ] **Step 11: Commit**

```bash
git add macd_divergence_screener_v5.py tests/test_app.py
git commit -m "Grade scan signals with higher timeframe, OI, BTC regime and the lower-timeframe trigger"
```

---

### Task 5: Chart preview endpoint

**Files:**
- Modify: `macd_divergence_screener_v5.py` (imports ~L102; new `render_chart` before `# ── State persistence`; new route after `signals_route`)
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: `_charts`, `signals_for_tf` output (`_df`, `_d`), `TF_CFG`, `DISPLAY_TZ_OFFSET_H`.
- Produces: `render_chart(df, d, sig) -> bytes` (PNG); `GET /chart?pair=&side=&tf=` → `image/png`, or 404 with a plain-text reason.

- [ ] **Step 1: Install matplotlib**

Run: `pip install matplotlib`
Expected: installs (or "Requirement already satisfied").

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_app.py` (above the `Page` class), plus `from tests.helpers import LONG, wave_df  # noqa: E402` under the existing `import macd_divergence_screener_v5 as m` line:

```python
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
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `python -m unittest tests.test_app -v`
Expected: `no attribute 'render_chart'`, and `/chart` returns 404 for the known row.

- [ ] **Step 4: Imports**

Change the import block at the top of the file:

```python
import ccxt, pandas as pd, numpy as np, json, time, threading, queue, datetime, os, io
import re, statistics, urllib.request, urllib.error
from matplotlib.figure import Figure   # no pyplot: Figure alone is safe across Flask threads
```

- [ ] **Step 5: Add `render_chart`**

Directly above the `# ── State persistence ──` comment, add:

```python
# ── Chart preview ──────────────────────────────────────────────────────────────
def render_chart(df, d, sig):
    """PNG of the A-to-B leg: candles with the stop and R lines over the MACD
    panel with its trough and crossover dots. `d` is the raw detection (bar
    indexes into `df`), `sig` the finished signal (levels)."""
    lo = max(0, d["a_start"] - 10)
    v = df.iloc[lo:min(len(df), d["signal_bar"] + 30)]
    x = np.arange(len(v))
    fig = Figure(figsize=(9, 5.5), tight_layout=True)
    ax, am = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    up = (v["close"] >= v["open"]).values
    for mask, col in ((up, "#26a69a"), (~up, "#ef5350")):
        ax.vlines(x[mask], v["low"].values[mask], v["high"].values[mask], color=col, lw=1)
        ax.vlines(x[mask], v["open"].values[mask], v["close"].values[mask], color=col, lw=3)
    ax.plot([d["a_p"] - lo, d["b_p"] - lo], [d["a_price"], d["b_price"]],
            color="#f5a623", lw=1.5, marker="o")
    t = sig["targets"]
    for lvl, col, lbl in ((sig["stop_price"], "#ef5350", "stop"), (t["1r"], "#26a69a", "1R"),
                          (t["2r"], "#26a69a", "2R"), (t["3r"], "#26a69a", "3R")):
        ax.axhline(lvl, color=col, lw=0.8, ls="--")
        ax.text(len(v) - 1, lvl, lbl, color=col, fontsize=8, va="bottom", ha="right")
    ax.set_title(f"{sig['base']} {TF_CFG[sig['tf']]['label']} {d['side']} divergence, "
                 f"grade {sig['grade']} ({sig['score']})", fontsize=10)

    am.plot(x, v["macd"].values, color="#2962ff", lw=1.2, label="MACD")
    am.plot(x, v["signal"].values, color="#ff6d00", lw=1, label="signal")
    am.axhline(0, color="#888888", lw=0.5)
    am.plot([d["a_m"] - lo, d["b_m"] - lo], [d["a_macd"], d["b_macd"]],
            color="#f5a623", lw=1.5, marker="o")
    for bar, direction in d["crosses"]:
        am.plot(bar - lo, v["signal"].values[bar - lo], "o", ms=4,
                color="#26a69a" if direction > 0 else "#ef5350")
    am.legend(loc="upper left", fontsize=7)
    step = max(1, len(v) // 6)
    shown = v.index[::step] + pd.Timedelta(hours=DISPLAY_TZ_OFFSET_H)
    am.set_xticks(x[::step])
    am.set_xticklabels([ts.strftime("%b %d %H:%M") for ts in shown], fontsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    return buf.getvalue()
```

- [ ] **Step 6: Add the route**

Directly after `signals_route`, add:

```python
@app.route("/chart")
def chart_route():
    hit = _charts.get("|".join(request.args.get(k, "") for k in ("pair", "side", "tf")))
    if hit is None:
        return "No chart data for this row. Run the scan again.", 404
    return Response(render_chart(*hit), mimetype="image/png")
```

- [ ] **Step 7: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 8: Look at one chart**

Run the app, scan a few coins until a row appears, then open `http://localhost:5099/chart?pair=<PAIR>&side=<side>&tf=<tf>` with the values of that row from `/signals` (URL-encode the pair, e.g. `ENS%2FUSDT%3AUSDT`).
Expected: candles on top with an orange A-B line, a red dashed stop and three green dashed R lines; MACD and signal below with an orange line between the two trough dots and a dot on every crossover from A's start to the signal bar. Stop the server.

- [ ] **Step 9: Commit**

```bash
git add macd_divergence_screener_v5.py tests/test_app.py
git commit -m "Add the chart preview endpoint"
```

---

### Task 6: Dashboard table, detail row and help text

All edits are inside the `HTML = r"""…"""` string of `macd_divergence_screener_v5.py`.

**Files:**
- Modify: `macd_divergence_screener_v5.py` (CSS ~L2476; `strHint` ~L2635; results `<thead>` ~L2693; Learn page ~L2783-2905; JS: after `qcell`, `volCell`, `ctxCell`, `TIP`, `STR_HINTS`, `tgtHtml`, `detailRow`, `buildRow`, `renderAll`)
- Modify: `tests/test_app.py`

**Interfaces:**
- Consumes: row JSON from `/signals` and the `result` SSE event. Row: `pair, base, side, tfs, signals{tf: sig}, primary_tf, score, grade, regime, status, entry, weekly_volume, mover, flow, context_error`. Sig: `grade, score, regime, status, forming, signal_price, signal_time, bars_ago, stop_price, risk_pct, targets{1r,2r,3r,structure}, targets_from, entry_ref_price, trigger_price, pivot1_time, pivot2_time, pivot1_price, pivot2_price, pivot1_macd, pivot2_macd, pct_chg, tags`. `GET /chart?pair&side&tf`.

- [ ] **Step 1: Write the failing test**

In `tests/test_app.py`, replace `Page.test_new_table_and_filters` with:

```python
    def test_new_table_and_filters(self):
        for want in ('data-f="enter"', 'data-f="both"', "function evCell", "function ctxCell",
                     'id="scan-age"', 'id="dash-cal"',
                     "function gradeCell", "/chart?pair=", "'W-PATTERN'", "'LTF TRIGGER'"):
            self.assertIn(want, self.html, want)

    def test_pivot_era_ui_is_gone(self):
        for gone in ("macd_cross", "volCell", "FROM CURRENT PRICE", "MACD pivot must align"):
            self.assertNotIn(gone, self.html, gone)
```

Run: `python -m unittest tests.test_app -v` — expected: both fail.

- [ ] **Step 2: CSS**

After the line `tr.det b{color:var(--tx)}` add:

```css
tr.det img.chart{display:block;width:100%;max-width:900px;margin:8px 0;border:1px solid var(--bd2);border-radius:4px}
```

- [ ] **Step 3: Table header**

Replace the whole `<thead>…</thead>` of `<table id="resT">` with:

```html
        <thead><tr>
          <th>Pair</th>
          <th class="sortable sorted" data-sort="score" title="A checklist, not a prediction. First matching rule wins. No: the stop was hit, price already ran 1R, the signal bar is older than 6 bars on 4H or 12 on 1H, or the grade is C. Not yet: counter trend below grade A, price more than 2% past the signal price, a market-mover print inside 24 hours, or a signal bar still forming with no lower-timeframe trigger. Otherwise Yes. Default order: Yes, then Not yet, then No; highest score first, No rows newest first.">Good to enter? &#8595;</th>
          <th title="Regular bullish: across two MACD troughs, price makes a lower low while the MACD line makes a higher trough. Regular bearish is the mirror on two peaks. The tags show which timeframes diverge; the highlighted one supplies the answer and the levels.">Side &middot; TF</th>
          <th title="Quality score 0-100 and its grade: A is 70 or more, B is 55-69, C is below 55. It describes the setup; it is not a win rate. Run the backtest to see how each grade has done.">Grade</th>
          <th title="WITH TREND: BTC 4H RSI and this coin's 200 EMA both agree with the side. COUNTER TREND setups need grade A to answer Yes. Below it: what price has done since the signal bar.">Regime &middot; Status</th>
          <th class="sortable" data-sort="age" title="Close of the signal bar: the MACD cross that ends the second wave.">Signal</th>
          <th title="Beyond price extreme B by 0.25 ATR. Price trading through it means the divergence has failed.">Stop</th>
          <th title="Multiples of the distance from the entry reference to the stop. The entry reference is the signal price, or the lower-timeframe trigger price when that fired first. The structure target is the price at the top of the bump between A and B.">Targets (1R / 2R / 3R)</th>
          <th title="Times of the price extremes inside wave A and wave B.">A &rarr; B</th>
          <th title="Never part of the answer. Pattern tags (TRIPLE, W-PATTERN, LTF TRIGGER, ZERO-LINE CROSS), then the daily-movers and open-interest tags. Hover a tag for its definition.">Tags</th>
        </tr></thead>
```

- [ ] **Step 4: Strictness hints**

Replace the text inside `<div class="hint" id="strHint">…</div>` with:

`Two MACD waves with a real bump between them; the price and MACD differences must clear a minimum size.`

Replace the whole `var STR_HINTS = {…};` statement with:

```js
var STR_HINTS = {
  balanced: 'Two MACD waves with a real bump between them; the price and MACD differences must clear a minimum size.',
  strict:   'Larger minimum size, and both troughs must sit below the zero line (both peaks above it for bearish).'
};
```

- [ ] **Step 5: Grade cell, tags, tooltips**

Directly after the `qcell` function add:

```js
function gradeCell(p){
  var c = p.grade === 'A' ? 'var(--bull)' : p.grade === 'B' ? 'var(--fresh)' : 'var(--tx3)';
  return '<div class="qc"><span class="qnum" style="color:' + c + '">' + p.grade +
    '<span style="font-size:9px;opacity:.7;margin-left:3px">' + p.score + '</span></span>' +
    '<div class="qbar"><div class="qfill" style="width:' + p.score + '%;background:' + c + '"></div></div></div>';
}
```

Delete the `volCell` function.

Replace the whole `ctxCell` function with:

```js
function ctxCell(r){
  var h = [];
  (primary(r).tags || []).forEach(function(f){
    h.push('<span class="wflag wf-hioi"' + tip(f) + '>' + f + '</span>');
  });
  if(r.context_error) h.push('<span class="ptm" title="' + calEsc(r.context_error) + '">context unavailable</span>');
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
```

In the `TIP` object, change the last entry's line to end with a comma and add four entries:

```js
  'BELOW 7D LOW':    'Price has traded through the 7-day low.',
  'TRIPLE':          'The trough before A also diverges from A: three troughs in a row, price lower each time, MACD higher each time.',
  'W-PATTERN':       'The bump between A and B is at least half the MACD range of the leg: a clear W (M for bearish).',
  'LTF TRIGGER':     'The lower timeframe (1H for a 4H signal, 15m for a 1H signal) has crossed its MACD and closed beyond its last swing since B.',
  'ZERO-LINE CROSS': 'The MACD line has already crossed zero since the signal bar.'
```

- [ ] **Step 6: Targets, detail row, main row**

Replace the whole `tgtHtml`, `detailRow` and `buildRow` functions with:

```js
function tgtHtml(d){
  var tgt = d.targets || {}, trig = d.targets_from === 'trigger';
  return '<div style="font-family:var(--mono);font-size:8px;color:' + (trig ? 'var(--fresh)' : 'var(--tx3)') + ';margin-bottom:2px">' +
    (trig ? 'FROM TRIGGER PRICE ' + fmtPx(d.entry_ref_price) : 'FROM SIGNAL PRICE') + '</div>' +
    '<span style="color:var(--tx3)">' + fmtPx(tgt['1r']) + '</span>' +
    ' <span style="color:var(--tx3)">/</span> <span style="color:var(--tx2)">' + fmtPx(tgt['2r']) + '</span>' +
    ' <span style="color:var(--tx3)">/</span> <span style="color:var(--bull);font-weight:600">' + fmtPx(tgt['3r']) + '</span>' +
    '<div style="font-size:8px;color:var(--tx3);margin-top:2px">structure ' + fmtPx(tgt.structure) + '</div>';
}
function detailRow(r){
  var tr = document.createElement('tr');
  tr.className = 'det';
  var h = r.tfs.map(function(tf){
    var s = r.signals[tf], c = SC[s.status] || SC.active;
    return '<div><b>' + tfLabel(tf) + '</b> &nbsp;A ' + s.pivot1_time + ' @ ' + fmtPx(s.pivot1_price) +
      ' &rarr; B ' + s.pivot2_time + ' @ ' + fmtPx(s.pivot2_price) +
      ' &nbsp;&middot;&nbsp; MACD ' + s.pivot1_macd + ' &rarr; ' + s.pivot2_macd +
      ' &nbsp;&middot;&nbsp; grade ' + s.grade + ' (' + s.score + ') &middot; ' + s.regime +
      ' &nbsp;&middot;&nbsp; ' + c.lbl + ' (' + (s.pct_chg >= 0 ? '+' : '') + s.pct_chg + '% in signal direction)' +
      (s.trigger_price != null ? ' &nbsp;&middot;&nbsp; lower-timeframe trigger at ' + fmtPx(s.trigger_price) : '') +
      ' &nbsp;&middot;&nbsp; stop ' + fmtPx(s.stop_price) + ' (risk ' + s.risk_pct + '%)' +
      ' &nbsp;&middot;&nbsp; 1R ' + fmtPx((s.targets || {})['1r']) + ' / 2R ' + fmtPx((s.targets || {})['2r']) +
      ' / 3R ' + fmtPx((s.targets || {})['3r']) + '</div>' +
      '<img class="chart" loading="lazy" alt="Chart needs a fresh scan" src="/chart?pair=' +
      encodeURIComponent(r.pair) + '&side=' + r.side + '&tf=' + tf + '">';
  }).join('');
  tr.innerHTML = '<td colspan="10"><div>Weekly volume ' + fmtVol(r.weekly_volume) + ' (7 days)</div>' + h + '</td>';
  return tr;
}
function buildRow(r){
  var bull = r.side === 'bullish', p = primary(r), tf = r.primary_tf || r.tfs[0];
  var tr = document.createElement('tr');
  tr.className = 'main-row ' + (bull ? 'rb' : 'rr') + (verdict(r) === 'skip' ? ' dim' : '');
  var tfs = r.tfs.map(function(t){ return '<span class="tfb' + (t === tf ? ' tfb-on' : '') + '">' + tfLabel(t) + '</span>'; }).join('');
  tr.innerHTML =
    '<td><div class="pb"><span class="pb-pair">' + r.base + '</span><span class="pb-q">USDT PERP</span></div></td>' +
    '<td>' + evCell(r) + '</td>' +
    '<td><span class="bdg ' + (bull ? 'bb' : 'br') + '">' + (bull ? 'BULLISH' : 'BEARISH') + '</span>' + tfs + '</td>' +
    '<td>' + gradeCell(p) + '</td>' +
    '<td><div class="ptm" style="color:var(--tx);font-size:11px">' + p.regime + '</div><div class="ptm">' + (SC[p.status] || SC.active).lbl + '</div></td>' +
    '<td><span style="font-family:var(--mono);font-size:11px">' + fmtPx(p.signal_price) + '</span><div class="ptm">' +
      (p.forming ? tfLabel(tf) + ' &middot; bar still forming' : p.signal_time + ' &middot; ' + b2t(p.bars_ago, tf)) + '</div></td>' +
    '<td><span style="font-family:var(--mono);font-size:11px;color:var(--bear)">' + fmtPx(p.stop_price) + '</span><div style="font-family:var(--mono);font-size:9px;color:var(--tx3)">risk ' + p.risk_pct + '%</div></td>' +
    '<td style="font-family:var(--mono);font-size:10px">' + tgtHtml(p) + '</td>' +
    '<td><div class="ptm">A ' + p.pivot1_time + '</div><div class="ptm">B ' + p.pivot2_time + '</div></td>' +
    '<td>' + ctxCell(r) + '</td>';
  tr.addEventListener('click', function(){
    var n = tr.nextSibling;
    if(n && n.className === 'det'){ n.parentNode.removeChild(n); return; }
    tr.parentNode.insertBefore(detailRow(r), tr.nextSibling);
  });
  return tr;
}
```

- [ ] **Step 7: Row order**

In `renderAll`, replace the `rows.sort(function(a, b){ … });` statement with:

```js
  rows.sort(function(a, b){
    if(sortKey === 'age') return ageMin(a) - ageMin(b);
    return (EV_RANK[verdict(a)] - EV_RANK[verdict(b)]) ||
           (verdict(a) === 'skip' ? ageMin(a) - ageMin(b) : 0) ||   /* played-out rows: newest first */
           (b.score - a.score);
  });
```

- [ ] **Step 8: Learn page**

(a) In the first paragraph of `<div id="pg-learn">`, replace the sentence `Between pivot A and pivot B the price line and the MACD line move in opposite directions; that mismatch is the whole signal.` with:

`It is read the way you would on a chart: the MACD/signal crossovers split the MACD line into waves, each wave below the signal line is one trough, and two neighbouring troughs A and B are compared with the price lows inside those same waves. Price lower, MACD trough higher: that mismatch is the whole signal.`

(b) Replace the whole `<div class="dash-sec" style="margin-top:28px">` block titled `The rules a signal has to pass (Balanced)` with:

```html
  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">The rules a signal has to pass (Balanced)</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Completed candles only.</b> The still-forming bar is dropped before anything is calculated, so two scans inside the same bar agree. The one exception is a row marked &ldquo;bar still forming&rdquo;: it is provisional and can disappear.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Waves come from the crossovers.</b> A bear wave runs from a bearish MACD/signal cross to the next bullish cross, a bull wave the other way. A cross only counts if the two lines separate by 5% of the recent MACD range within 3 bars, and a wave shorter than 3 bars is merged into its neighbours, so a tiny double dot cannot split one trough into two.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>One trough per wave.</b> The MACD trough is the lowest MACD-line value in the wave. The price low is the lowest wick from the start of the wave to 2 bars after it ends, because price often bottoms just after momentum turns.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>A then B, with a real bump between.</b> Two neighbouring troughs, 8 to 80 bars apart. The MACD peak between them must rise at least 25% of the leg's MACD range above the higher trough: that is the middle of the W.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Minimum size.</b> The price difference must clear both a fixed floor and a fraction of ATR, and the MACD difference must be at least 15% of the leg's MACD range. This is what kills micro-divergences.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>Reported only when wave B is closed.</b> The cross that ends wave B is the signal bar and its close is the signal price. Before that cross there is no signal.</div></div>
      <div class="rule"><div class="rule-n">07</div><div class="rule-b"><b>Liquidity gate.</b> Rolling 7-day traded value must clear the floor you set. Measured from the candles as a ratio against the exchange's own 24h figure, so it does not depend on guessing whether a venue reports volume in coins, contracts or dollars.</div></div>
    </div>
  </div>
```

(c) Replace the whole `<div class="dash-sec" style="margin-top:28px">` block titled `Entry verdict &mdash; a checklist, not a prediction` with:

```html
  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">Good to enter? &mdash; a checklist, not a prediction</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>No comes first.</b> The stop was hit, price already made 1R or more, the signal bar is older than 6 bars on 4H or 12 on 1H, or the grade is C.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Not yet names the one thing missing.</b> A counter-trend setup below grade A (watch only), a pullback to the signal price after price ran more than 2%, a CPI / FOMC-class print inside 24 hours, or a signal bar that has not closed and has no lower-timeframe trigger.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Yes (confirmed) and Yes (early).</b> Confirmed means the signal bar has closed and nothing above applies. Early means the signal bar is still forming but the lower timeframe (1H for 4H, 15m for 1H) has already crossed its MACD and closed beyond its last swing since B.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Regime changes the bar, it never blocks.</b> WITH TREND means BTC 4H RSI is on the signal's side of 50 and the coin is on the right side of its 200 EMA. Anything else is COUNTER TREND and needs grade A.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>The answer ages.</b> It uses the live price at the moment of the scan. Once the scan is more than an hour old the answers dim and the page asks for a re-run.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>One row per coin and side.</b> When 4H and 1H diverge the same way the row shows both tags and takes its answer, grade and levels from the better one (Yes beats Not yet beats No, 4H wins a tie); the highlighted tag shows which. Click a row for each timeframe's levels and its chart.</div></div>
    </div>
  </div>
```

(d) Replace the whole last `<div class="dash-sec">` block titled `Setup score &mdash; only used to order rows` with:

```html
  <div class="dash-sec">
    <div class="dash-sec-ttl">Quality score and grade</div>
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75">
      0 to 100, graded <b style="color:var(--tx)">A</b> (70+), <b style="color:var(--tx)">B</b> (55-69) or
      <b style="color:var(--tx)">C</b> (below 55). It adds up: the price difference A to B in ATR (15), the MACD
      trough improvement (15), the height of the bump (10), both troughs on the classic side of zero (10), the
      higher timeframe's MACD agreeing (15), volume drying up in wave B (10), a rejection wick at B (10), open
      interest falling into B with funding not against it (10), a triple divergence (5). Open interest rising
      more than 5% against the signal costs 10.
      <br><br>
      The grade describes the setup. It is not a win rate. To see how each grade and regime has actually done on
      the stored candles, run <b style="color:var(--tx)">python macd_divergence_screener_v5.py --backtest</b>.
    </div>
  </div>
```

In the Movers section, rule 07, replace `the same caveat as the setup score` with `it is not a win rate`.

- [ ] **Step 9: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass (including `test_no_unfilled_template_placeholders`).

- [ ] **Step 10: Check the page by hand**

Run the app and scan. Check: ten columns line up with their headers; the grade shows a letter with the score beside it; a row click opens the detail line with a chart image under each timeframe; clicking again closes it; the *Signal* header sorts newest first; filters still work; after restarting the server the table is restored and the chart area shows "Chart needs a fresh scan". Stop the server.

- [ ] **Step 11: Commit**

```bash
git add macd_divergence_screener_v5.py tests/test_app.py
git commit -m "Show grade, regime, tags and the chart preview in the divergence table"
```

---

### Task 7: Backtest mode

**Files:**
- Modify: `macd_divergence_screener_v5.py` (new section directly above `# ── State persistence`; `if __name__ == "__main__":` block)
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `detect_wave_divergence`, `_mk`, `score_signal`, `regime_tag`, `htf_agrees`, `asof`, `with_indicators`, `usdt_perps`, `fetch_btc_rsi`, `fetch_df_deep`, `drop_forming_candle`, `RateLimiter`, `make_exchange`, `usd_volume`.
- Produces:
  - `BT_WARMUP = 80`.
  - `trade_outcome(df, i, side, entry, stop) -> {"hit": 0..3, "stopped": bool, "bars": int|None, "mae_r": float}`.
  - `backtest_pair(df, tf, rules, htf=None, btc=None) -> list[trade]`; a trade is the outcome dict plus `tf, side, score, grade, regime, time`.
  - `summarise(trades, key) -> list[dict]` with `key, n, win1, avg_r1, win2, avg_r2, win3, avg_r3`.
  - `run_backtest(exid="okx", n_pairs=30, symbols=(), strictness="balanced")` prints the report.
  - CLI: `python macd_divergence_screener_v5.py --backtest [--exchange okx] [--pairs 30] [--symbols BTC,ETH] [--strictness balanced]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
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


class Outcome(unittest.TestCase):
    def test_stop_is_checked_before_the_target_inside_one_bar(self):
        df = pd.DataFrame({"high": [100, 140.0], "low": [100, 80.0]})
        self.assertEqual(m.trade_outcome(df, 0, "bullish", 100.0, 90.0),
                         {"hit": 0, "stopped": True, "bars": 1, "mae_r": 2.0})

    def test_short_reaching_3r(self):
        df = pd.DataFrame({"high": [100, 101, 100.0], "low": [100, 79, 69.0]})
        self.assertEqual(m.trade_outcome(df, 0, "bearish", 100.0, 110.0),
                         {"hit": 3, "stopped": False, "bars": 2, "mae_r": 0.1})


class Replay(unittest.TestCase):
    def test_signal_is_taken_on_the_bar_it_first_exists(self):
        t = m.backtest_pair(frame(), "1h", RULES)
        self.assertEqual(len(t), 1)
        self.assertEqual((t[0]["side"], t[0]["tf"], t[0]["time"]),
                         ("bullish", "1h", str(wave_df(REPLAY).index[126])))
        self.assertEqual((t[0]["hit"], t[0]["stopped"], t[0]["bars"]), (0, False, None))
        self.assertEqual((t[0]["score"], t[0]["grade"], t[0]["regime"]), (53, "C", "COUNTER TREND"))

    def test_outcome_follows_the_bars_after_entry(self):
        win = m.backtest_pair(frame(high=130.0), "1h", RULES)[0]
        self.assertEqual((win["hit"], win["stopped"], win["bars"]), (3, False, 4))
        loss = m.backtest_pair(frame(low=80.0), "1h", RULES)[0]
        self.assertEqual((loss["hit"], loss["stopped"], loss["bars"]), (0, True, 4))

    def test_no_lookahead_what_happens_later_cannot_change_the_signal(self):
        keys = ("time", "side", "score", "grade", "regime")
        a = m.backtest_pair(frame(), "1h", RULES)[0]
        b = m.backtest_pair(frame(high=130.0, low=80.0), "1h", RULES)[0]
        self.assertEqual([a[k] for k in keys], [b[k] for k in keys])


class Summary(unittest.TestCase):
    def test_win_rate_and_average_r_per_group(self):
        trades = [{"grade": "A", "hit": 3, "stopped": False},
                  {"grade": "A", "hit": 1, "stopped": True},
                  {"grade": "A", "hit": 0, "stopped": False},      # still open: in n, not in the rates
                  {"grade": "B", "hit": 0, "stopped": True}]
        self.assertEqual(m.summarise(trades, "grade"), [
            {"grade": "A", "n": 3, "win1": 100.0, "avg_r1": 1.0,
             "win2": 50.0, "avg_r2": 0.5, "win3": 50.0, "avg_r3": 1.0},
            {"grade": "B", "n": 1, "win1": 0.0, "avg_r1": -1.0,
             "win2": 0.0, "avg_r2": -1.0, "win3": 0.0, "avg_r3": -1.0}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `python -m unittest tests.test_backtest -v`
Expected: errors, `no attribute 'trade_outcome'` / `'backtest_pair'` / `'summarise'`.

- [ ] **Step 3: Add the backtest**

Directly above the `# ── State persistence ──` comment (below `render_chart`), add:

```python
# ── Backtest ───────────────────────────────────────────────────────────────────
BT_WARMUP = 80      # bars before the replay starts, so MACD has settled


def trade_outcome(df, i, side, entry, stop):
    """What happened to a trade entered at the close of bar i.

    hit = highest R target (0-3) reached before the stop. Inside one bar the
    stop is checked first, the conservative read when a bar touches both.
    """
    k = 1 if side == "bullish" else -1
    risk = max((entry - stop) * k, entry * 0.005)
    hi, lo = df["high"].values, df["low"].values
    hit, mae, stopped, bars = 0, 0.0, False, None
    for j in range(i + 1, len(df)):
        worst, best = (lo[j], hi[j]) if k > 0 else (hi[j], lo[j])
        mae = max(mae, (entry - worst) * k / risk)
        if (worst - stop) * k <= 0:
            stopped, bars = True, j - i
            break
        hit = max(hit, min(3, int((best - entry) * k // risk)))
        if hit == 3:
            bars = j - i
            break
    return {"hit": hit, "stopped": stopped, "bars": bars, "mae_r": round(float(mae), 2)}


def backtest_pair(df, tf, rules, htf=None, btc=None):
    """Replay one pair bar by bar. Bar i sees df.iloc[:i + 1] and nothing else.

    A signal is taken the first time the detector reports it, at that bar's
    close. Not replayed: open interest (no bar-by-bar history) and the
    lower-timeframe trigger, so scores here run up to 10 points below live.
    """
    cfg, seen, trades = TF_CFG[tf], set(), []
    close = df["close"].values
    for i in range(BT_WARMUP, len(df) - 1):
        view = df.iloc[:i + 1]
        for side in ("bullish", "bearish"):
            k = 1 if side == "bullish" else -1
            for d in detect_wave_divergence(view, side, rules):
                key = (side, d["b_m"])
                if d["signal_bar"] <= i - WHIP_BARS or key in seen:
                    continue                       # old news, or already taken
                seen.add(key)
                sig = _mk(view, d, cfg)
                entry, stop = float(close[i]), sig["stop_price"]
                if (entry - stop) * k <= 0:
                    continue                       # already through the stop when first seen
                close_ts = view.index[-1] + pd.Timedelta(minutes=cfg["minutes"])
                btc_rsi = asof(btc["rsi"], close_ts - pd.Timedelta(minutes=240)) \
                    if btc is not None else None
                score, grade = score_signal(sig, htf_agrees(htf, close_ts, cfg["htf_minutes"], side))
                trades.append(dict(
                    trade_outcome(df, i, side, entry, stop), tf=tf, side=side,
                    score=score, grade=grade, time=str(view.index[-1]),
                    regime=regime_tag(side, entry > float(view["ema200"].iloc[-1]), btc_rsi)))
    return trades


def summarise(trades, key):
    """Per group: count, and for each target R the win rate (%) and average R
    of exiting there, over trades that reached that target or the stop."""
    rows = []
    for g in sorted({t[key] for t in trades}):
        ts = [t for t in trades if t[key] == g]
        row = {key: g, "n": len(ts)}
        for r in (1, 2, 3):
            done = [t for t in ts if t["stopped"] or t["hit"] >= r]
            wins = sum(t["hit"] >= r for t in done)
            row[f"win{r}"] = round(100 * wins / len(done), 1) if done else None
            row[f"avg_r{r}"] = round((wins * r - (len(done) - wins)) / len(done), 2) if done else None
        rows.append(row)
    return rows


def run_backtest(exid="okx", n_pairs=30, symbols=(), strictness="balanced"):
    """Fetch the scan's own history for the most liquid pairs and print win
    rate and average R per grade and per regime."""
    rules = STRICTNESS[strictness]
    ex = make_exchange(exid)
    mkts = ex.load_markets()
    limiter = RateLimiter(SPEED["safe"]["rps"][exid])
    perps = usdt_perps(mkts)
    if symbols:
        pairs = [s for s in perps if s.split("/")[0] in symbols]
    else:
        tickers = ex.fetch_tickers()
        pairs = sorted(perps, key=lambda s: usd_volume(tickers.get(s)) or 0, reverse=True)[:n_pairs]

    def load(sym, tf, bars, minutes):
        df = drop_forming_candle(fetch_df_deep(ex, limiter, sym, tf, bars), minutes)
        return with_indicators(df) if df is not None and len(df) >= 80 else None

    btc = fetch_btc_rsi(ex, limiter)
    trades = []
    for n, sym in enumerate(pairs, 1):
        d4 = load(sym, "4h", TF_CFG["4h"]["bars"], 240)
        d1h = load(sym, "1h", TF_CFG["1h"]["bars"], 60)
        d1d = load(sym, "1d", HTF_BARS_1D, 1440)
        got = (backtest_pair(d4, "4h", rules, d1d, btc) if d4 is not None else []) + \
              (backtest_pair(d1h, "1h", rules, d4, btc) if d1h is not None else [])
        trades += got
        print(f"[{n}/{len(pairs)}] {sym}: {len(got)} signal(s)")

    def cell(v, suffix=""):
        return f"{'-' if v is None else str(v) + suffix:>9}"

    for key in ("grade", "regime"):
        print(f"\nBy {key}")
        print(f"{key:<15}{'n':>5}" + "".join(f"{f'win@{r}R':>9}{f'avgR@{r}R':>9}" for r in (1, 2, 3)))
        for row in summarise(trades, key):
            print(f"{row[key]:<15}{row['n']:>5}" +
                  "".join(cell(row[f'win{r}'], '%') + cell(row[f'avg_r{r}']) for r in (1, 2, 3)))
    print("\nwin@NR = share of resolved trades that reached N x risk before the stop; "
          "avgR@NR = average result of exiting there.\n"
          "Entry is the close of the bar the signal first appears on. Open interest and the "
          "lower-timeframe trigger are not replayed,\nso grades here can sit up to 10 points "
          "below the live scan's.")
```

- [ ] **Step 4: Command line**

Replace the `if __name__ == "__main__":` block at the end of the file with:

```python
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MACD wave divergence screener")
    ap.add_argument("--backtest", action="store_true",
                    help="replay the stored history and print win rates instead of starting the dashboard")
    ap.add_argument("--exchange", default="okx", choices=sorted(ALLOWED_EXCHANGES))
    ap.add_argument("--pairs", type=int, default=30, help="backtest the N most liquid pairs")
    ap.add_argument("--symbols", default="", help="backtest exactly these, e.g. BTC,ETH")
    ap.add_argument("--strictness", default="balanced", choices=sorted(STRICTNESS))
    args = ap.parse_args()
    if args.backtest:
        run_backtest(args.exchange, args.pairs,
                     {s.strip().upper() for s in args.symbols.split(",") if s.strip()},
                     args.strictness)
    else:
        load_dotenv()     # JBLANKED_API_KEY from .env, if there is one
        load_state()      # restore the last scan so the dashboard isn't amnesiac
        load_movers()     # ...and the last movers scan
        load_calendar()   # ...and the news calendar
        app.run(debug=False, threaded=True, port=5099)
```

- [ ] **Step 5: Run the tests**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 6: Run a real backtest**

Run: `python macd_divergence_screener_v5.py --backtest --exchange binance --symbols BTC,ETH,SOL`
Expected: three `[n/3] …: N signal(s)` lines (a few seconds each), then a "By grade" and a "By regime" table with numbers, then the two-line note. No traceback.

- [ ] **Step 7: Commit**

```bash
git add macd_divergence_screener_v5.py tests/test_backtest.py
git commit -m "Add a bar-by-bar backtest mode reporting win rate and R per grade and regime"
```

---

### Task 8: Module docstring and README

**Files:**
- Modify: `macd_divergence_screener_v5.py` (module docstring, L1-100)
- Modify: `README.md`

- [ ] **Step 1: Module docstring**

Change the first line to:

`"""MACD Divergence Screener v5 — 4H + 1H regular divergences read on MACD crossover waves`

Directly under it (before `WHAT CHANGED IN v5`), insert:

```
WAVE DETECTION (spec v2, docs/superpowers/specs/2026-11-19-macd-wave-divergence-v2.md)
──────────────────────────────────────────────────────────────────────────────────────
Divergence is no longer found from price pivots. The MACD/signal crossovers
split the MACD line into waves; each wave below the signal line is one trough
(above it, one peak); two neighbouring troughs A and B are compared with the
price lows inside those same waves (detect_wave_divergence). A signal exists
only once the cross that ends wave B has printed: that bar is the signal bar.
Each signal gets a 0-100 quality score and an A/B/C grade (score_signal), a
WITH TREND / COUNTER TREND tag from BTC 4H RSI and the coin's 200 EMA, and an
optional early entry from the lower timeframe (ltf_trigger). `--backtest`
replays the same detector bar by bar (backtest_pair) and reports win rate and
R per grade and regime. Everything below about pivots, PIVOT_SYNC, the clean
line and the setup score describes the detector this replaced.
```

- [ ] **Step 2: README**

Replace the intro paragraph's first sentence fragment `for **regular MACD divergences** (bullish and bearish) on the **4H and 1H** timeframes in one run` with `for **regular MACD divergences** (bullish and bearish), read trough to trough on MACD crossover waves, on the **4H and 1H** timeframes in one run`.

Replace the bullets **Divergence detection**, **Strictness presets**, **Setup score, only for ordering**, **One row per coin and side** and **Entry verdict** with:

```markdown
- **Divergence detection**: the MACD/signal crossovers split the MACD line
  into waves. Each wave below the signal line is one trough (above it, one
  peak). Two neighbouring troughs A and B, 8 to 80 bars apart with a real
  bump between them, are compared with the lowest wicks inside those same
  waves: price lower, MACD trough higher is a regular bullish divergence
  (bearish is the mirror). A cross that never opens a gap and a wave shorter
  than 3 bars are merged away, so a double dot cannot split a trough. A
  divergence is reported only once the cross that ends wave B has closed;
  that bar is the signal bar. The still-forming candle is dropped, with one
  labelled exception (see *Good to enter?*).
- **Strictness presets**: `balanced` (default) and `strict`. Strict needs a
  larger divergence and both troughs below the zero line (peaks above it).
- **Quality score and grade**: 0-100, graded A (70+), B (55-69) or C. Points
  for the size of the price and MACD differences, the height of the bump,
  the zero-line side, the higher timeframe's MACD agreeing (1D for a 4H
  signal, 4H for a 1H signal), volume drying up in wave B, a rejection wick
  at B, open interest falling into B, and a triple divergence. It describes
  the setup; the backtest below is what measures it.
- **Regime tag**: `WITH TREND` when BTC 4H RSI is on the signal's side of 50
  and the coin is on the right side of its 200 EMA, else `COUNTER TREND`. It
  never hides a row; a counter-trend setup needs grade A to answer Yes.
- **One row per coin and side**: when 4H and 1H diverge the same way they
  share a row, which takes its answer, grade and levels from the better
  signal (Yes beats Not yet beats No, 4H wins a tie); the highlighted tag
  shows which. Click a row for each timeframe's levels and a chart of the
  A-to-B leg with the troughs, crossovers, stop and 1R / 2R / 3R drawn in.
- **Good to enter?**: every row answers Yes, Not yet or No in plain language.
  First matching rule wins. No: stop hit, already ran 1R, signal bar older
  than 6 bars (4H) or 12 bars (1H), or grade C. Not yet: counter trend below
  grade A, price more than 2% past the signal price, or a market-mover print
  inside 24 hours. Otherwise `Yes (confirmed)`. A signal bar that is still
  forming is listed as provisional: `Yes (early)` if the lower timeframe (1H
  for 4H, 15m for 1H) has already crossed its MACD and closed beyond its last
  swing since B, otherwise Not yet. It is a checklist judged at scan time,
  not a prediction; the page dims answers once the scan is over an hour old.
```

In **Context tags**, append: ` Pattern tags sit in the same column: TRIPLE, W-PATTERN, LTF TRIGGER, ZERO-LINE CROSS.`

Replace the Requirements list and command with:

```markdown
- Python 3.9+
- `flask`, `ccxt`, `pandas`, `numpy`, `matplotlib` (chart preview)

```bash
pip install flask ccxt pandas numpy matplotlib
```
```

After the `## Running` section (before `## API`), add:

```markdown
## Backtest

```bash
python macd_divergence_screener_v5.py --backtest                      # 30 most liquid pairs on OKX
python macd_divergence_screener_v5.py --backtest --exchange binance --pairs 60
python macd_divergence_screener_v5.py --backtest --symbols BTC,ETH,SOL --strictness strict
```

Replays the same candles the scan uses (720 bars of 4H, 500 of 1H) bar by
bar with the same detector, each bar seeing only the bars before it. A
signal is entered at the close of the bar it first appears on, with the
scan's stop. The report gives, per grade and per regime, how many signals
there were and, for 1R, 2R and 3R targets, the share that got there before
the stop and the average R of exiting there. Use it to check whether grade A
actually beats grade B before trusting the cutoffs.

Limits: open interest and the lower-timeframe trigger are not replayed (no
bar-by-bar history), so backtest grades can sit up to 10 points below live
ones; when one bar touches both the stop and a target, the stop is counted.
```

In the API table add the row:

`| `/chart` | GET | PNG of one row's A-to-B leg from the last scan (`pair`, `side`, `tf`) |`

In **Reading a row**: in item 1 replace `` reads `WAIT` `` with `answers Not yet`; replace items 3 and 4 with:

```markdown
3. **Start with the Yes rows, highest grade first.** Click a row and check
   the chart: the two troughs, the bump between them and the stop.
4. **Not yet rows are the ones to keep an eye on.** The reason says what to
   wait for: a pullback to the signal price, the macro print, the signal bar
   closing, or (counter trend, grade B) nothing: watch only.
```

Replace the Disclaimer text with: `The grade describes setup quality; the backtest measures the past, not the future. This is a research/screening tool, not trading advice.`

- [ ] **Step 3: Full test run**

Run: `python -m unittest discover -s tests -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add macd_divergence_screener_v5.py README.md
git commit -m "Document wave detection, grades and the backtest"
```

---

## Self-review notes

Spec coverage: Steps 1-4 → Task 1. Step 5, stop/targets, Step 6 score, Step 7 regime, status, "Good to enter?", rows → Task 2. Step 6 context components and Step 8 → Tasks 3-4. Indicators and history sizes → Tasks 1 and 4. Chart preview → Tasks 5-6. Columns and new tags → Task 6. Backtest → Task 7. Universe, Movers, Calendar → untouched.

Not built, on purpose: a UI switch for the extended two-bump mode (`MAX_BUMPS` is a constant), a backtest tab in the dashboard (command line only), chart data that survives a restart.
