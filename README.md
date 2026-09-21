# trading-divergence-screener

A self-contained Flask app that scans linear USDT perpetuals on OKX, Binance,
and Bybit for **regular MACD divergences** (bullish and bearish), plus two
companion scans — an open-interest flow watchlist and a daily-movers scan.
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
- Multi-threaded scanning (per-thread `ccxt` instance, shared token-bucket
  rate limiter) with `safe` / `normal` / `fast` speed presets per exchange.

## Requirements

- Python 3.9+
- `flask`, `ccxt`, `pandas`, `numpy`

```bash
pip install flask ccxt pandas numpy
```

## Running

```bash
python macd_divergence_screener_v5.py
```

Serves on `http://localhost:5099`. Open `/` in a browser for the dashboard,
which drives the scans below via Server-Sent Events and shows live progress,
results, and any partial/incomplete-scan warnings.

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

Common query params (all optional, sane defaults apply):

- `exchange` — `okx` (default), `binance`, or `bybit`
- `tf` — timeframe for `/scan`: `4h` (default) or `1d`
- `strictness` — `loose`, `balanced` (default), `strict` (`/scan` only)
- `speed` — `safe`, `normal` (default), `fast`
- `min_weekly_vol` — liquidity floor in USD (default `70000000`)
- `max_pairs` — cap on pairs scanned (default `400`)
- `symbols` — comma-separated symbol allowlist (`/scan` only)

## Disclaimer

Signal scores describe setup quality, not a backtested edge. This is a
research/screening tool, not trading advice.
