# trading-divergence-screener

A self-contained Flask app that scans linear USDT perpetuals on OKX, Binance,
and Bybit for **regular MACD divergences** (bullish and bearish), read trough to trough on MACD crossover waves, on the **4H
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
- **All exchanges (default)**: one row per coin, Binance first. A coin
  Binance does not list is read on whichever of OKX / Bybit has the larger
  24h volume and tagged `NOT ON BINANCE`. Each row's primary timeframe is
  then re-detected on the other venues listing the coin; where the same side
  fires within one bar the row is tagged `ALSO ON OKX, BYBIT` (only the
  venues that confirmed). A tag, not points; a venue that fails to fetch
  never costs the row. The venue the data came from is shown under the pair.
  Picking one exchange scans only that one, without venue tags.
- **Liquidity gate**: a pair is only scanned if its true rolling 7-day traded
  value clears a floor (default $70M/week ≈ $10M/day).
- **Thin-coin filters**: a top-of-book spread above 0.10% or a 4H ATR outside
  0.6%–8% of price keeps a pair out of the scan; a timeframe with fewer than
  300 completed bars is not analysed on that timeframe; a pair with a setup
  but under $5M open interest (latest daily OI point) has its rows dropped.
  Unknown spread or OI never filters, a symbol override skips all four, and
  the audit line counts each. There's no trend/regime gate, since regular
  divergences are counter-trend by definition.
- **Divergence detection**: the MACD/signal crossovers split the MACD line
  into waves. Each wave below the signal line is one trough (above it, one
  peak). Two neighbouring troughs A and B, 8 to 80 bars apart with a real
  bump between them, are compared with the lowest wicks inside those same
  waves: price lower, MACD trough higher is a regular bullish divergence
  (bearish is the mirror). A cross that never opens a gap (5% of the 50-bar
  MACD range and 0.05 ATR, within 3 bars) and a wave shorter
  than 3 bars are merged away, so a double dot cannot split a trough. A
  divergence is reported only once the cross that ends wave B has closed;
  that bar is the signal bar. The still-forming candle is dropped, with one
  labelled exception (see *Good to enter?*).
- **Strictness presets**: `balanced` (default) and `strict`. Strict needs a
  larger divergence and both troughs below the zero line (peaks above it).
- **Quality score and grade**: 0-100, graded A (70+), B (55-69) or C.
  Pattern points (up to 70) for the size of the price and MACD differences,
  the height of the bump, the zero-line side, volume drying up in wave B, a
  rejection wick at B and a triple divergence. Context points (10 each) for
  the higher timeframe's MACD agreeing (1D for a 4H signal, 4H for a 1H
  signal), open interest falling into B with funding against the crowd, and
  the BTC 4H state on the signal's side; OI rising more than 5% against the
  signal costs 10. Context the data cannot check is left out and the score
  rescaled to 100 (tag `PARTIAL DATA`). It describes the setup; the backtest
  below is what measures it.
- **Regime tag**: BTC 4H is BULL (RSI > 55 and close above its 200 EMA),
  BEAR (RSI < 45 and close below it) or CHOP. `WITH TREND` when the BTC state
  and the coin's own 200 EMA both agree with the side, `CHOP` when BTC is
  CHOP, else `COUNTER TREND` (also when BTC data is missing). It never hides
  a row; a counter-trend setup needs grade A to answer Yes, a CHOP reversal
  keeps the normal cutoff.
- **Transparent scan accounting**: every run reports pairs dispatched,
  analysed, rejected on volume, errored, or missing volume data, and whether
  the scan actually completed — so a quiet-looking result can't be mistaken
  for "the market is quiet" when it was actually a partial/failed scan.
- **One row per coin and side**: when 4H and 1H diverge the same way they
  share a row, which takes its answer, grade and levels from the better
  signal (Yes beats Not yet beats No, 4H wins a tie); the highlighted tag
  shows which. Click a row for each timeframe's levels and a chart of the
  A-to-B leg with the troughs, crossovers, stop and 1R / 2R / 3R drawn in.
- **Good to enter?**: every row answers Yes, Not yet or No in plain language.
  First matching rule wins. No: stop hit, already ran 1R, signal bar closed
  more than 24 hours ago (4H) or 8 hours ago (1H), or grade C. Not yet:
  counter trend below grade A, price more than 2% past the signal price, or
  a high-impact USD print (CPI, Core PCE, PPI, non-farm payrolls, unemployment
  rate, FOMC rate / statement / press conference, Fed Chair) due within 12
  hours for a 4H signal or 4 hours for a 1H signal, or printed in the last
  hour (`MACRO_BLOCKERS`, `MACRO_WAIT_H`). The other market movers never
  block; one due within 24 hours shows as a `MACRO <name> in Xh` tag.
  Otherwise `Yes (confirmed)`. A signal bar that is still
  forming is listed as provisional: `Yes (early)` if the lower timeframe (1H
  for 4H, 15m for 1H) has already crossed its MACD and closed beyond its last
  swing since B, otherwise Not yet. A 4H signal also checks 15m (tag
  `15M TRIGGER`, counted only if it closes within 8 hours of the 4H close);
  the trigger price is whichever trigger closed first. It is a checklist judged at scan time,
  not a prediction; the page dims answers once the scan is over an hour old.
- **Context tags**: for coins that have a divergence, the scan also reads the
  last completed day against the coin's 20-day baseline (volume spike, range
  expansion, at the 7-day high or low) and 7-day open-interest flow with
  funding (new longs, crowded longs, squeeze fuel, spot-led). Tags never
  change the answer, and neither does `MACRO` (a non-blocking market mover
  due within 24 hours). Pattern tags sit in the same column: TRIPLE,
  W-PATTERN, LTF TRIGGER, 15M TRIGGER, ZERO-LINE CROSS.
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
- `flask`, `ccxt`, `pandas`, `numpy`, `matplotlib` (chart preview)

```bash
pip install flask ccxt pandas numpy matplotlib
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

## Backtest

```bash
python macd_divergence_screener_v5.py --backtest                      # 30 most liquid pairs on OKX
python macd_divergence_screener_v5.py --backtest --exchange binance --pairs 60
python macd_divergence_screener_v5.py --backtest --symbols BTC,ETH,SOL --strictness strict
```

Replays the same candles the scan uses (720 bars of 4H, 500 of 1H) bar by
bar with the same detector, each bar seeing only the bars up to and including it. A
signal is entered at the close of the bar it first appears on, with the
scan's stop. The report gives, per grade and per regime, how many signals
there were and, for 1R, 2R and 3R targets, the share that got there before
the stop and the average R of exiting there. Use it to check whether grade A
actually beats grade B before trusting the cutoffs.

Limits: open interest and the lower-timeframe trigger are not replayed (no
bar-by-bar history); the OI component is left out of the backtest score and
the rest rescaled, as the live scan does when OI is missing. When one bar
touches both the stop and a target, the stop is counted.

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
| `/chart` | GET | PNG of one row's A-to-B leg from the last scan (`pair`, `side`, `tf`) |
| `/calendar` | GET | Upcoming economic events, cached; `refresh=1` forces a re-fetch (at most every 5 min) |

Common query params (all optional, sane defaults apply):

- `exchange` — `all` (default for `/scan` and `/universe`: Binance first, see
  above), `okx`, `binance`, or `bybit`. `/movers` takes one exchange (default `okx`)
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

1. **Check the calendar strip first.** A high-impact USD print within 12
   hours holds every otherwise-ready 4H row at Not yet (within 4 hours, the
   1H rows too), and so does one printed in the last hour; the reason names
   the event. The other movers show as a `MACRO` tag on the row instead.
2. **Read the audit line.** A short list from an unfinished scan is not a
   quiet market.
3. **Start with the Yes rows, highest grade first.** Click a row and check
   the chart: the two troughs, the bump between them and the stop.
4. **Not yet rows are the ones to keep an eye on.** The reason says what to
   wait for: a pullback to the signal price, the macro print, the signal bar
   closing, or (counter trend, grade B) nothing: watch only.
5. **Weigh the context tags.** A bearish divergence `AT RANGE HIGH` with
   `CROWDED LONGS` is a stronger fade than the same divergence mid-range.
   `SQUEEZE FUEL` under a bearish row is a warning. `HIGH OI/VOL` means size
   down.
6. **Re-run before acting on an old table.** Answers use the price at scan
   time.

The *Specific coins* box scans only the tickers you type and skips the
volume floor and the thin-coin filters, so a thin name can still be checked
on demand. Use the exchange with the deepest book (Binance for most pairs),
or All.

## Disclaimer

The grade describes setup quality; the backtest measures the past, not the future. This is a research/screening tool, not trading advice.
