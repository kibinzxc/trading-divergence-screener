# trading-divergence-screener

A self-contained Flask app that scans linear USDT perpetuals on OKX, Binance,
and Bybit for **regular MACD divergences** (bullish and bearish), plus two
companion scans — an open-interest flow watchlist and a daily-movers scan —
and a news calendar of the scheduled macro prints that move crypto.
Everything runs from one file: `macd_divergence_screener_v5.py` serves both
the API and a browser dashboard.

## What it does

- **Universe**: every active linear `.../USDT:USDT` perp on the chosen
  exchange, minus stablecoin bases and TradFi perps (stocks/ETFs/commodities
  listed as perps — filtered via exchange instrument metadata, with a name
  backstop). No hardcoded coin basket.
- **Liquidity gate**: a pair is only scanned if its true rolling 7-day traded
  value clears a floor (default $70M/week ≈ $10M/day). This is the only
  universe filter — there's no trend/regime gate, since regular divergences
  are counter-trend by definition.
- **Divergence detection**: price pivots are found on real swing highs/lows
  (not closes), matched to a MACD pivot within a few bars, required to have a
  "clean" line between the two pivots, and sized against both a percentage
  floor and ATR/MACD-range fractions to filter out micro-divergences. The
  still-forming candle is always dropped so nothing repaints between scans.
- **Strictness presets**: `loose`, `balanced` (default), `strict` — control
  how much of the above is enforced and whether a zero-line condition is
  required. See the module docstring in the script for the full rationale.
- **Setup score, not a win-rate**: each signal gets a score describing how
  textbook the divergence structure is. It is not a backtested probability.
- **Transparent scan accounting**: every run reports pairs dispatched,
  analysed, rejected on volume, errored, or missing volume data, and whether
  the scan actually completed — so a quiet-looking result can't be mistaken
  for "the market is quiet" when it was actually a partial/failed scan.
- **OI flow watchlist** (`/watchlist`): a lighter, ~7-day open-interest vs.
  price-change scan that classifies pairs into quadrants (e.g. spot-led,
  crowded-long, short squeeze fuel) using funding rate as a leverage signal.
- **Daily movers** (`/movers`): a cheap, ~1-call-per-pair scan for pairs
  moving significantly off their recent volume/range baseline.
- **News calendar** (`/calendar`): the scheduled macro prints that actually
  move crypto perps — US CPI/PCE/PPI, the jobs report, FOMC and Powell, GDP,
  retail sales, ISM, plus the BoJ/ECB/BoE rate decisions — on a two-week grid
  with a countdown to the next one. Everything else on the economic calendar
  is hidden unless you ask for it. Data from ForexFactory's weekly feed,
  extended beyond the current week by JBlanked (ForexFactory's calendar
  behind a free API key). See [News calendar](#news-calendar).
- Multi-threaded scanning (per-thread `ccxt` instance, shared token-bucket
  rate limiter) with `safe` / `normal` / `fast` speed presets per exchange.

## Requirements

- Python 3.9+
- `flask`, `ccxt`, `pandas`, `numpy`

```bash
pip install flask ccxt pandas numpy
```

Optional: a [JBlanked](https://www.jblanked.com/news/api/docs/) API key to
extend the news calendar past the current week. Their calendar endpoints are
metered — each call spends an account credit — so the screener fetches them
only every 6 hours, about 4 credits a day. Without a key, or with an account
out of credits, the calendar still works from ForexFactory's free weekly
feed; that feed is the authoritative source for the current week anyway.

## Running

```bash
python macd_divergence_screener_v5.py
```

Serves on `http://localhost:5099`. Open `/` in a browser for the dashboard,
which drives the scans below via Server-Sent Events and shows live progress,
results, and any partial/incomplete-scan warnings.

To extend the calendar beyond the current week, put your JBlanked key in a
`.env` file next to the script (the file is gitignored):

```
jblanked_api_key=YOUR_KEY
```

The script reads `.env` itself at startup; no extra package and no shell
export needed. A real environment variable `JBLANKED_API_KEY` also works.

## API

All scan endpoints stream progress via SSE (`log`, `progress`, `result`,
`error`, `done` events) and cache their last completed result, retrievable
without re-scanning.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/scan` | GET (SSE) | Run a MACD divergence scan |
| `/signals` | GET | Last cached `/scan` result |
| `/watchlist` | GET (SSE) | Run an OI-flow scan |
| `/watch_signals` | GET | Last cached `/watchlist` result |
| `/movers` | GET (SSE) | Run a daily-movers scan |
| `/mover_signals` | GET | Last cached `/movers` result |
| `/universe` | GET | Size of the tradable universe at a given volume floor (no scan) |
| `/calendar` | GET | Upcoming economic events, cached; `refresh=1` forces a re-fetch (at most every 5 min) |

Common query params (all optional, sane defaults apply):

- `exchange` — `okx` (default), `binance`, or `bybit`
- `tf` — timeframe for `/scan`: `4h` (default) or `1d`
- `strictness` — `loose`, `balanced` (default), `strict` (`/scan` only)
- `speed` — `safe`, `normal` (default), `fast`
- `min_weekly_vol` — liquidity floor in USD (default `70000000`)
- `max_pairs` — cap on pairs scanned (default `400`)
- `symbols` — comma-separated symbol allowlist (`/scan` only)

## News calendar

The Calendar tab answers one question: is there a scheduled print inside the
next day or two that can blow through a stop? It is deliberately narrow.

**What counts as a market mover.** A curated list, not the feed's own impact
rating — ForexFactory marks Australian jobs data and RBA speeches as High,
and those do nothing to crypto. The default view shows only:

- US inflation: CPI, Core CPI, Core PCE, PPI, Core PPI
- US jobs report: Non-Farm Employment Change, Unemployment Rate, Average
  Hourly Earnings
- Fed: Federal Funds Rate, FOMC Statement / Press Conference / Minutes /
  Projections, Fed Chair speeches and testimony
- US growth: Advance / Prelim / Final GDP, Retail Sales, Core Retail Sales,
  ISM Manufacturing and Services PMI
- Other central bank rate decisions only: BoJ, ECB, BoE

Weekly jobless claims, JOLTS, consumer confidence, flash PMIs, other Fed
speakers, every non-US data print and bank holidays are excluded on purpose.
The *All high-impact* toggle shows everything the feed rates High. The list
is `CAL_MOVERS` in the script: one currency plus one regex per rule.

**Sources.** ForexFactory's weekly JSON is authoritative for the current
week: it is free, complete, its times carry real UTC offsets and its numbers
have units. JBlanked only extends the horizon past that week. Its copy drops
some events (High-impact ones included), reports `0` where it has no value,
and runs on a broker clock, so it is aligned to the ForexFactory week rather
than trusted over it, and it supplies the actual value once a print is out.

The two are fetched on different clocks because they cost different things.
ForexFactory is free and refreshes every 30 minutes. JBlanked spends an
account credit per call and only carries next week's schedule, which barely
changes, so it refreshes every 6 hours and its last pull is cached (on disk,
so a restart does not re-spend). If it fails, the cached horizon stays and
the reason is named in the audit line. A 401 from JBlanked is usually an
account with no credits left rather than a bad key; the audit line says
which. Both caches live in `calendar_state.json`. The Refresh button
re-fetches at most once every five minutes.

**Times.** Shown in your browser's local time, with UTC in the tooltip. The
script measures JBlanked's clock against the shared ForexFactory events on
every refresh, corrects it, and reports the shift in the audit line. If
ForexFactory is unreachable there is nothing to measure against, so the last
known offset is reused and the audit line says so.

**The Dashboard strip** repeats the next market mover with a countdown and
turns amber when it is inside 24 hours. `/#calendar` opens the tab directly.

## Daily routine

The three scans answer three different questions. Movers finds what moved,
the Scanner finds where momentum disagrees with price, and the Watchlist says
whether new money is behind it. Run them in that order once a day. The same
guide lives in the dashboard under the **Routine** tab.

**Before you run anything**

1. Run after the daily close. Perp daily candles close at 00:00 UTC (08:00
   Manila). Every scan reads the last *completed* daily bar, so aim for
   08:00 to 10:00 Manila.
2. Use the exchange with the deepest book (Binance for most pairs). Some
   majors trade thinly elsewhere and get volume-rejected for no chart reason.
3. The Min 7-day volume box on the Scanner tab applies to all three scans.
   Default `$70M/week` is roughly `$10M/day`. Names under it do not exist to
   the screener.
4. Read the audit line under every table. A short list from an unfinished
   scan is not a quiet market.
5. Check the Calendar tab. A fresh entry sized into a CPI or FOMC print is a
   coin flip, not a setup. If the next market mover is inside 24 hours, wait
   for it or size for it.

**Step 1 — Movers: what did something unusual yesterday**

- Keep the *All* filter and read by score. *Top 4* is a shortcut, not the list.
- Extended runs: `ABOVE 7D HIGH` / `AT RANGE HIGH` plus a big 1-day change.
  Treat as fade-or-wait candidates, not chases. Breakdowns are the mirror.
- Every row carries PDH, PDL, the 7-day range and where price sits in it.
  Write the level down.
- Blind spot: a coin that had a quiet day is dropped even if it is in a clean
  consolidation after a breakout. Step 2 covers that.

**Step 2 — Scanner on 4H: where momentum disagrees with price**

- Run 4H on *Balanced*. On the Signals tab look at *Fresh* (second pivot
  within the last 12 bars) and *At the signal* (price still within ~2% of
  the signal). Those are the entry candidates.
- *Ran* and *invalidated* are history, not entries.
- Bullish rows are pullback candidates. Bearish rows on names that Step 1
  flagged as extended are fade candidates.
- Then run 1D for context (*Fresh* there means within 3 bars).
- The Symbols box scans only the tickers you type and skips the volume
  floor, so a thin name can still be checked on demand.

**Step 3 — Watchlist (OI flow): is new money behind it**

- Ranked by 7-day OI change; reads best on a Sunday. Use it to qualify names
  from Steps 1 and 2, not to find new ones.
- Extended run: you want `NEW LONGS` + `SPOT-LED`. `CROWDED LONGS` makes a
  fade stronger.
- Breakdown: `NEW SHORTS` + `SQUEEZE FUEL` is the warning; a pullback long
  beats a short there.
- `HIGH OI/VOL` means violent moves either way. Size down.

**Step 4 — Build the shortlist**

- Three to five names, split into *extended* (wait for a fade signal or a
  reclaim) and *pullback* (fresh bullish divergence or a held prior-day low).
- Each name needs a level, a stop and a target before the session: PDH/PDL
  and the 7-day range from Movers, the stop and 1R/2R/3R from the Scanner.
- Scores rank, they do not predict. Neither score is a backtested win rate.

**Step 5 — Compare with a mentor's watchlist**

- For each posted name, note which tab surfaced it, and if none did, why:
  volume-rejected, quiet day, or a real miss. The audit line answers the
  first one.
- Volume-rejected is by design. Decide once whether you want thin names.
- A real miss is a clear structure no scan flagged. If the same kind of miss
  repeats, the screener needs a new flag, not a lower floor.

## Disclaimer

Signal scores describe setup quality, not a backtested edge. This is a
research/screening tool, not trading advice.
