# One scan, one table: 4H and 1H MACD divergences

Date: 2026-11-19
Status: awaiting review

## Problem

The screener exists to find coins with a MACD divergence on 4H or 1H. Today
that purpose is spread over eight tabs and three separate scans, there is no
1H timeframe, and the divergence list has two trust problems reported by the
user:

1. **Can't see it on the chart.** Some listed divergences are not visible as
   a MACD-line divergence on TradingView.
2. **Too late.** Divergences stay listed for days after they formed, mixed in
   with ones that already ran or were invalidated.

## Goals

- One Run button and one results table whose rows are divergences.
- Scan 4H and 1H in the same run. Drop 1D.
- Movers and OI flow survive only as context badges on divergence rows.
- List a divergence sooner, and stop listing it once it is old.
- Reject divergences whose line is too long or too shallow to see.
- Say, per row, whether the setup is enterable as of the scan.

## Non-goals

- No backtest, no win rate. The score stays a structural description.
- No histogram divergence. The user reads the MACD line, which is what
  `detect_divergences` already measures.
- No chart rendering in the app.
- No change to the Calendar tab or its data sources.
- A coin that moved or has an OI shift but no divergence is no longer listed
  anywhere. This is an accepted loss.

## Design

### 1. Pages

Three tabs: **Scan**, **Calendar**, **Method**.

- **Scan** replaces Dashboard, Scanner and Signals. Top to bottom: the
  next-market-mover strip (unchanged), settings, Run button and progress,
  filter row, results table, audit line.
- **Calendar** is unchanged.
- **Method** is kept, with its text updated to match this spec.
- Removed: the Watchlist, Movers and Routine tabs and all their markup, CSS
  and JavaScript.

Settings kept: exchange, min 7-day volume (and "How many coins pass?"),
strictness (Balanced, Strict), max pairs, speed, symbols box.
Settings removed: timeframe toggle, Loose strictness, thin-coins toggle.

### 2. Scan pipeline (`/scan`)

Per pair, in the existing worker pool:

1. Fetch 4H candles, drop the forming candle, measure 7-day volume. If the
   pair fails the volume floor, stop. No 1H call is spent on it.
2. Fetch 1H candles, drop the forming candle.
3. Run `detect_divergences` on each timeframe with that timeframe's config,
   then `evaluate_status`, `dedupe`, `build_targets`, `setup_score`, as today.
4. If the pair has at least one divergence, fetch context:
   - **Movers context:** daily candles, then the existing `analyse_mover` and
     `classify_mover`.
   - **OI context:** OI history, then the existing `classify_flow`, using a
     funding map fetched once per scan with `fetch_funding_map`.
   The per-pair bodies of `movers_route` and `watchlist_route` are extracted
   into two functions so the scan can call them. Nothing about how they
   classify changes.
5. A context fetch that fails does not drop the row. The row carries
   `context_error` and the table shows "context unavailable".

The `tf` query parameter is removed. `strictness=loose` falls back to
`balanced`.

Cost: two candle calls per pair that clears the pre-gate (one if it fails the
volume floor), plus roughly two calls for each pair that has a divergence.

### 3. Row shape

Signals are grouped by `(pair, side)` into one row:

```
{
  "pair", "base", "side", "current_price", "weekly_volume",
  "tfs": ["4h", "1h"],           # which timeframes diverge, 4h first
  "primary_tf": "4h" | "1h",
  "signals": {"4h": {...}, "1h": {...}},   # existing signal records
  "score": max of the signals' scores,
  "fresh": any signal fresh,
  "status": status of the primary signal,
  "entry": {"verdict": "enter" | "wait" | "skip", "reason": str},
  "mover": {...} | null,         # classify_mover output
  "flow": {...} | null,          # classify_flow output
  "context_error": str | null
}
```

The **primary signal** is the one with the best entry verdict (enter, then wait, then skip), 4H winning ties; the row records it as `primary_tf`. (Changed after final review: always preferring 4H hid fresh 1H setups behind a played-out 4H signal.) The
table's age, stop and 1R/2R/3R columns show the primary signal. Expanding a
row shows the pivots and levels of every timeframe in the row.

Sort order: `enter` rows first, then `wait`, then `skip`; within each, rows
with both timeframes first, then by score.

Filters: **Live** (default: verdict `enter` or `wait`), **Enter now**,
**Bullish**, **Bearish**, **4H+1H**, **All**. `skip` rows appear only under
All.

### 3a. Entry verdict

A pure function `entry_verdict(sig, hours_to_event)` run over each signal; the row shows the best one,
where `hours_to_event` is the time to the next curated market mover from the
calendar cache (`None` if unknown). The first matching rule wins:

| # | Condition | Verdict | Reason shown |
|---|---|---|---|
| 1 | status `invalidated` | skip | Stop was hit |
| 2 | status `ran` | skip | Already ran N R |
| 3 | not `fresh` | skip | Too old (N bars) |
| 4 | status `extended` | wait | Moved N% from signal, wait for a pullback to P |
| 5 | no MACD cross since the second pivot | wait | Momentum has not turned, wait for the MACD cross |
| 6 | market mover within 24 hours | wait | EVENT in N h |
| 7 | otherwise | enter | Near signal, MACD crossed, stop S (risk N%) |

So `enter` means all of: second pivot within the fresh window, price within
2% of the signal price (`MOVED_PCT`), stop not touched, MACD crossed its
signal line after the pivot, and no curated macro print inside 24 hours (the
same threshold that turns the calendar strip amber).

The verdict is a checklist, not a prediction, and the Method tab says so. It
is computed at scan time from the live price at that moment. The table shows
the scan's age next to the verdict column and dims the verdicts once the scan
is older than one bar of the fastest timeframe (1 hour), with a prompt to
re-run.

OI flow and Movers badges do not change the verdict. They stay as context for
the user to weigh.

### 4. Timeliness

`find_pivots` gains a separate right-side window. A pivot must still be the
extreme of `lb` bars to its left, but only `rb` bars to its right. It stays
non-repainting: a pivot is confirmed once `rb` closed bars follow it, and
never moves afterwards.

| Timeframe | `pivot_lb` | `pivot_rb` | `min_dist` | `max_dist` | `max_age` | `fresh` | bars fetched |
|---|---|---|---|---|---|---|---|
| 4H (today) | 5 | 5 | 5 | 120 | 60 | 12 | 720 |
| 4H (new) | 5 | 3 | 5 | 60 | 30 | 12 | 720 |
| 1H (new) | 5 | 3 | 5 | 60 | 24 | 12 | 500 |

Effect: a 4H divergence can be listed 12 hours after its second pivot
instead of 20, a 1H one after 3 hours, and neither is listed past 5 days
(4H) or 1 day (1H).

MACD pivots use the same `lb`/`rb` as price pivots. `PIVOT_SYNC` stays 3.

### 5. Visibility

- `max_dist` drops from 120 to 60 bars on 4H (table above), so a line never
  spans more than 10 days.
- Balanced `min_macd_frac` rises from 0.08 to 0.15: the MACD pivots must
  differ by at least 15% of the MACD range across the leg.
- The clean-line and MACD-pivot rules are already on in Balanced and stay on.

**These two numbers are a best guess.** No failing examples were supplied, so
they are not tuned against a real miss. The implementation logs signal counts
before and after on one live scan so the effect is visible. If the user later
names coins and dates where a listed divergence was not visible, those become
regression tests and the numbers are tuned against them.

### 6. Removed

- Routes: `/watchlist`, `/watch_signals`, `/movers`, `/mover_signals`.
- The thin band (`MOVER_THIN_MIN_WEEKLY_VOL` and the `thin` parameter).
- `MOVERS_STATE_FILE`, `save_movers`, `load_movers`, and the watchlist's
  cached state.
- `TF_CFG["1d"]` and `STRICTNESS["loose"]`.
- Tests that cover only removed behaviour.

Kept and reused: `analyse_mover`, `classify_mover`, `classify_flow`,
`fetch_oi_history`, `fetch_funding_map`, `oi_point_usd`, and their tests.

### 7. Saved state

`/signals` returns the last result in the new row shape. A state file written
by the old version (flat signal list, has `meta.tf`) is ignored on load
rather than rendered, so the page starts empty instead of broken.

### 8. Docs

README and the module docstring are updated: new tab list, API table, query
parameters, and the Daily routine section replaced by a short "How to read a
row" section.

## Testing

- Existing calendar tests and the kept mover and flow tests pass unchanged.
- `find_pivots`: with `rb=3`, a pivot is found 3 bars after the extreme and
  is the same index when more bars are appended (no repaint).
- `detect_divergences`: a synthetic series with a clear bullish divergence is
  found with the 4H config and with the 1H config; the mirrored bearish
  series likewise.
- A divergence whose second pivot is older than `max_age` is not returned.
- A divergence whose pivots are more than `max_dist` apart is not returned.
- Row grouping: 4H and 1H signals on the same pair and side produce one row
  with `tfs == ["4h", "1h"]`; opposite sides produce two rows.
- `entry_verdict`: one test per rule in the table, plus rule order (an
  invalidated signal with a macro print inside 24 hours is `skip`, not
  `wait`) and `hours_to_event=None` not blocking `enter`.
- An old-shape state file loads as empty.
- Manual: run one live scan, confirm the audit line balances and that a row's
  pivots match the chart.

## Risks

- **Earlier confirmation adds pivots.** `rb=3` accepts swings that a 5-bar
  wait would have rejected. Mitigated by the higher `min_macd_frac` and
  measured by the before/after counts.
- **1H is noisier than 4H.** More rows, lower average quality. The 4H+1H
  filter and sort order put the agreeing setups first.
- **Scan time.** Two candle calls per pair as before, but 1H paginates more
  than 1D did for 500 bars on some exchanges.
