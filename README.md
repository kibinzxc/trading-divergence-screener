# trading-divergence-screener

A self-contained Flask app that scans linear USDT perpetuals on OKX, Binance,
and Bybit for **regular MACD divergences** (bullish and bearish) on the **4H
and 1H** timeframes in one run. Each result row says whether the setup is
enterable right now, and carries daily-movers and open-interest context for
that coin. A news calendar covers the scheduled macro prints that move
crypto. Everything runs from one file: `macd_divergence_screener_v5.py`
serves both the API and a browser dashboard.

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
  (not closes), matched to a MACD pivot within a few bars, confirmed once 3 closed bars follow it (5 bars before, 3 after), required to have a
  "clean" line between the two pivots, and sized against both a percentage
  floor and ATR/MACD-range fractions to filter out micro-divergences. The two pivots are at most 60 bars apart, and a divergence is never dropped for its age: each coin lists its most recent one per side and timeframe from the candles scanned (about 120 days of 4H, 21 days of 1H). The
  still-forming candle is always dropped so nothing repaints between scans.
- **Strictness presets**: `balanced` (default) and `strict`. Strict adds the
  zero-line condition and a larger minimum divergence.
- **Setup score, only for ordering**: each signal gets a score describing how
  textbook the divergence structure is. The table does not show it; it only
  breaks ties in the row order. It is not a backtested probability.
- **Transparent scan accounting**: every run reports pairs dispatched,
  analysed, rejected on volume, errored, or missing volume data, and whether
  the scan actually completed — so a quiet-looking result can't be mistaken
  for "the market is quiet" when it was actually a partial/failed scan.
- **One row per coin and side**: when 4H and 1H diverge the same way they
  share a row, which sorts first and takes its verdict, stop and targets
  from whichever signal is furthest along (Enter beats Wait beats Skip, 4H
  wins a tie); the highlighted tag shows which.
- **Entry verdict**: every row is `ENTER`, `WAIT` or `SKIP` with a plain-language
  answer to "is it good to enter now?". The table opens on All, so divergences
  that already played out stay listed (dimmed) under their 4H / 1H tags.
  `ENTER` means the second pivot is within 12 bars, price has not run more
  than 2% past the signal price, the stop is untouched, MACD has crossed its signal line since
  the pivot, and no market-mover print is due inside 24 hours. It is a
  checklist judged at scan time, not a prediction; the page dims verdicts
  once the scan is over an hour old.
- **Context tags**: for coins that have a divergence, the scan also reads the
  last completed day against the coin's 20-day baseline (volume spike, range
  expansion, at the 7-day high or low) and 7-day open-interest flow with
  funding (new longs, crowded longs, squeeze fuel, spot-led). Tags never
  change the verdict.
- **Movers tab** (`/movers`): a separate scan of every liquid crypto perp for
  an unusual last completed day (5%+ move, 2x its 20-day volume, or 1.5x its
  ATR20), whether or not the coin diverges. Each row lists the prior-day and
  7-day levels. An opt-in thin band adds coins from $20M a week up to the
  volume floor, held to volume spike and range expansion together.
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
which drives the scan below via Server-Sent Events and shows live progress,
results, and any partial/incomplete-scan warnings.

To extend the calendar beyond the current week, put your JBlanked key in a
`.env` file next to the script (the file is gitignored):

```
jblanked_api_key=YOUR_KEY
```

The script reads `.env` itself at startup; no extra package and no shell
export needed. A real environment variable `JBLANKED_API_KEY` also works.

## API

The scan endpoint streams progress via SSE (`log`, `progress`, `result`,
`error`, `done` events) and caches its last completed result, retrievable
without re-scanning.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/scan` | GET (SSE) | Run the 4H + 1H divergence scan |
| `/signals` | GET | Last cached `/scan` result (`rows`, `meta`, `ts`) |
| `/movers` | GET (SSE) | Run the daily movers scan; `thin=1` adds the low-liquidity band |
| `/mover_signals` | GET | Last cached `/movers` result (`rows`, `meta`, `ts`) |
| `/universe` | GET | Size of the tradable universe at a given volume floor (no scan) |
| `/calendar` | GET | Upcoming economic events, cached; `refresh=1` forces a re-fetch (at most every 5 min) |

Common query params (all optional, sane defaults apply):

- `exchange` — `okx` (default), `binance`, or `bybit`
- `strictness` — `balanced` (default) or `strict`
- `speed` — `safe`, `normal` (default), `fast`
- `min_weekly_vol` — liquidity floor in USD (default `70000000`)
- `max_pairs` — cap on pairs scanned (default `400`)
- `symbols` — comma-separated symbol allowlist

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

**Checking against the source.** The feed carries no actual values, only
forecast and previous, so the tab links out to ForexFactory in two places:
the header opens the calendar, and the selected day opens that same day
there, which is where the number appears once a print lands.

**The Scan-page strip** repeats the next market mover with a countdown and
turns amber when it is inside 24 hours. `/#calendar` opens the tab directly.

## Reading a row

1. **Check the calendar strip first.** If a market mover is inside 24 hours,
   every otherwise-ready row reads `WAIT` and names the event.
2. **Read the audit line.** A short list from an unfinished scan is not a
   quiet market.
3. **Start with `ENTER` rows, 4H+1H first.** Click a row for both
   timeframes' pivots and check the line on your chart before anything else.
4. **`WAIT` rows are the ones to keep an eye on.** The reason says what to wait for: the
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

## Disclaimer

Signal scores describe setup quality, not a backtested edge. This is a
research/screening tool, not trading advice.
