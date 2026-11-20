# Calm UI redesign (brief + shorter table)

Approved by the owner on 2026-11-20 from a clickable mockup. Goal: the Scan page should read
like a partner that says what matters now, instead of an 11-column table in 9-10px text.
Presentation only: detection, scoring, answers, filters and the backtest do not change.

## 1. The brief (new, top of the Scan page)
Built in the browser from the rows plus three new scan-meta fields (below). Plain sentences.
- Headline: greeting by local time, then one of: "N setups pass the checklist; the rest can
  wait." / "Nothing passes yet. N setups are close." / "Nothing passes the checklist right now.
  No trade is a fine answer."
- Market: "Market is bull / bear / in chop." with BTC's 4H close, 200 EMA and RSI, and one
  sentence on what the state means for entries (BULL: shorts are counter trend and need grade A;
  BEAR: mirror; CHOP: normal cutoff). BTC unavailable: say so, every setup is counter trend.
- Macro: the nearest USD blocker with the time left, computed live from its timestamp, and what
  waits on it (within 4h: all entries; within 12h: 4H only; otherwise nothing yet). None: "No
  major US release on the calendar." The nearest non-blocking mover within 24h gets one
  "context only" line.
- Picks, as clickable cards (coin, side, TF, grade and score, answer, the full reason, entry,
  stop with risk %, 2R, live price): up to 3 Yes rows, highest score first, labelled "Worth a
  look". With no Yes rows: up to 2 Not yet rows, "Closest to qualifying". With neither: the three
  newest signals, "Nothing to act on. The three newest signals, and why each is a No".
  Clicking a card opens that row in the table.
- A fixed line: passing the checklist is not a prediction; the backtest has not shown an edge.
- Before any scan the brief is the existing empty state (headline, one sentence, Scan button).

Scan meta gains (saved with the state; the page must cope when they are missing):
`btc`: `{state, rsi, close, ema200}` from the last completed BTC 4H bar, or null;
`blocker` and `macro`: `{ts_ms, title}` or null, from `macro_window()`.

## 2. The table
Seven columns: Coin, Side and TF, Grade, Answer, Entry, Stop, 7d OI.
- Coin cell: name, then at most two tags and "+N" for the rest.
- Answer cell: Yes / Not yet / No pill, and the first sentence of the reason under it.
- Stop cell: price and "risk x%". 7d OI stays sortable. Existing filter tabs keep working.
- No rows are folded under "Show N that did not pass, newest first" while the answer filter is
  on its default; picking a filter that asks for them shows them.
- Opened row: the chart, levels (now, entry, stop, 1R, 2R, 3R, structure target), setup (A time,
  B time, signal time, regime and status, exchange, timeframes), "Why this answer" (a tick /
  wait / cross per check: age against the limit, grade, regime, status, macro) with the full
  reason, and every tag with its existing tooltip. Whatever the opened row shows today stays
  reachable there.
- Rows open with click, Enter or Space.

## 3. The chart (existing matplotlib PNG, restyled)
Same candles and MACD panel. Add numbers: the entry line; prices on the entry, stop, 1R, 2R and
3R labels; A and B labelled with their price on the price panel and their MACD value on the MACD
panel. Dark colours matching the page.

## 4. The look (all five pages)
- Tokens from the mockup: background `#0C111B`, surfaces `#121926` / `#182132`, line `#222D40`,
  text `#DCE3EE` / `#9AA8BD` / `#6E7D94`, accent `#8FB4E8`, yes `#6FCF9F`, wait `#E2B866`,
  no `#8E9BB0`, short `#E58C98`.
- 15px base text, nothing below 12px. Inter for words; JetBrains Mono only for numbers.
- Green and red only for side and answer, amber only for waiting. One accent for buttons.
- Taller rows, fewer borders, visible keyboard focus, `prefers-reduced-motion` respected.
- The settings sidebar becomes a "Scan settings" panel on the Scan page that starts closed.
  Every control keeps its id and behaviour.
- Movers, Calendar, Tags and Method keep their layout and content; they take the new tokens,
  text sizes and spacing. Their text stays truthful about the columns that moved.

## Out of scope
Light theme, a phone layout (narrow screens scroll the table sideways), new data, any change to
detection, scoring or the answer rules.

## Constraints
One file (`macd_divergence_screener_v5.py`), no new dependencies, tests updated and green
(`python -m unittest discover -s tests`), the page script passes `node --check`, README kept
truthful, no `Co-Authored-By` or "Generated with Claude Code" lines, one commit, no push.

Reference mockup (same markup, CSS and brief logic to port):
`mock-template.html` in the session scratchpad; published copy
https://claude.ai/artifact/UuosECPfFKR9ULdM1MKeU6
