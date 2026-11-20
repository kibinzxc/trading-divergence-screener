#!/usr/bin/env python3
"""MACD Divergence Screener v5 — 4H + 1H regular divergences read on MACD crossover waves

WAVE DETECTION (spec v2, docs/superpowers/specs/2026-11-19-macd-wave-divergence-v2.md)
──────────────────────────────────────────────────────────────────────────────────────
Divergence is no longer found from price pivots. The MACD/signal crossovers
split the MACD line into waves; each wave below the signal line is one trough
(above it, one peak); two neighbouring troughs A and B are compared with the
price lows inside those same waves (detect_wave_divergence). A signal exists
only once the cross that ends wave B has printed: that bar is the signal bar.
Each signal gets a 0-100 quality score and an A/B/C grade (score_signal), a
WITH TREND / CHOP / COUNTER TREND tag from the BTC 4H state (RSI and 200 EMA,
btc_state) and the coin's 200 EMA, and an
optional early entry from the lower timeframe (ltf_trigger; a 4H signal also
checks 15m). `--backtest`
replays the same detector bar by bar (backtest_pair) and reports win rate and
expectancy after costs per timeframe, grade and regime, in and out of sample. Everything below about pivots, PIVOT_SYNC, the clean
line and the setup score describes the detector this replaced.

WHAT CHANGED IN v5
──────────────────
One scan instead of three. /scan reads 4H and 1H for every pair, groups the
signals into one row per coin and side, and attaches an entry verdict
(entry_verdict: a checklist, not a prediction). The OI-flow scan no longer
exists as a list; its classifier runs only for coins that have a divergence
and shows up as context tags, next to the daily-movers tags. The daily-movers
scan (/movers) keeps its own tab. 1D and the Loose preset are
gone. A pivot is confirmed 3 closed bars after it instead of 5, the two
pivots may be at most 60 bars apart, and Balanced needs the MACD pivots to
differ by 15% of the leg's range. Those last two numbers are a first guess,
not tuned against missed examples.

WHAT CHANGED FROM v3.3 (and why)
────────────────────────────────
1. WEEKLY EMA20 REGIME: REMOVED.
   It gated which signals were allowed to show. Regular divergences are
   counter-trend BY DEFINITION, so a trend filter mostly argued with the
   signal it was filtering. The weekly timeframe is now used for ONE thing:
   liquidity (see #2).

2. WEEKLY VOLUME GATE: NEW, and it is the only universe filter.
   A pair is scanned if its rolling 7-day traded value clears the floor
   (default $70M/week ≈ $10M/day). That is the "can I actually hold this
   for a week" test. See weekly_usd_volume() for how it is computed without
   trusting any exchange's OHLCV volume units.

3. RSI: REMOVED EVERYWHERE.
   No RSI30 timing gate, no RSI trigger-price alert, no 1H fetch (which also
   removes one API call per pair — a third of the old scan cost) (v5 brings 1H
   back as its own divergence timeframe.). Entry
   judgement is yours. What replaced it is mechanical and needs no second
   indicator: a signal is dead when price traded through the stop, and the
   screener reports how far it has run in R since the pivot.

4. HIDDEN DIVERGENCES: REMOVED. Only REGULAR BULLISH and REGULAR BEARISH.

5. BASKET / AVOID TIERS AND BACKTEST WIN RATES: REMOVED.
   Those numbers were measured on the v3 config — weekly-regime gate on,
   hidden divergences included, 22 fixed pairs. None of that is true here,
   so quoting 38.5% @2R next to these signals would be a lie with a
   confidence interval attached. The score is now a SETUP score: it
   describes how textbook the divergence structure is. It is not a win-rate
   estimate and has not been backtested.

6. THE "ONLY TOP 10 COINS" PROBLEM: FIXED.
   v3 defaulted to "smart" mode = a hardcoded 22-coin basket, so a fresh
   listing like ENS could never appear no matter how good the setup. There
   is no basket now. The scanner walks every active linear USDT perp on the
   exchange, keeps the ones above the weekly volume floor, and scans all of
   them (bounded by Max pairs, default 400).

7. SPEED: the scan is now threaded (per-thread ccxt instance, one shared
   token-bucket limiter) because scanning 300+ pairs sequentially at
   exchange rate limit would take most of an hour.

ACCURACY WORK (the actual point of the rewrite)
───────────────────────────────────────────────
v3 found price pivots and read MACD at those bars. That fires on any bar
where the numbers happen to disagree, including bars where MACD was not
making a swing at all. v4 requires the divergence to be a thing you could
draw:

  a. Pivots are found on real swing extremes — low for troughs, high for
     peaks — not on closes.
  b. The still-forming candle is dropped before anything is computed, so
     nothing repaints between scans.
  c. A price pivot only counts if a MACD pivot sits within PIVOT_SYNC bars
     of it, and the MACD value used is that pivot's own extreme. Momentum
     routinely turns a bar or two off price; requiring exact alignment
     misses real setups, requiring none accepts noise.
  d. The line has to be clean: no bar between A and B undercuts B's low
     (bullish) and no bar's MACD undercuts A's MACD low. If either happens,
     the swing you would have drawn isn't there.
  e. Minimum size: the price difference must clear both a floor and a
     fraction of ATR, and the MACD difference must be a meaningful share of
     the MACD range across the leg. Kills micro-divergences.
  f. The right pivot is compared against ALL prior pivots in the window
     (kept from v3.2 — this is what makes regular divergences show up at
     all), and among those that pass every test, the deepest MACD extreme
     wins, which is the line you would draw by hand.

Strictness is selectable. STRICT adds the classic zero-line condition
(both bullish MACD pivots below zero, both bearish above) and a larger
minimum divergence.

NOTHING GETS SILENTLY SKIPPED
─────────────────────────────
"3 live setups" must never be readable as "the market is quiet" when 40
coins failed to fetch. The scan now reports, per run: pairs dispatched,
pairs actually analysed, how many fell short on 7-day volume, how many
returned no usable candles, how many errored, how many the exchange gave
no 24h volume for at all, and whether the run finished. If the browser is
closed mid-scan the partial result is still saved — and flagged as
incomplete on the dashboard, because a partial scan silently overwriting a
complete one is how you end up trusting an empty table.
"""

from flask import Flask, Response, jsonify, request
from concurrent.futures import ThreadPoolExecutor
import ccxt, pandas as pd, numpy as np, json, time, threading, queue, datetime, os, io
import re, statistics, urllib.request, urllib.error
from matplotlib.figure import Figure   # no pyplot: Figure alone is safe across Flask threads

app = Flask(__name__)

# ── Indicator constants ────────────────────────────────────────────────────────
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_PERIOD  = 14
EMA_TREND   = 200
RSI_PERIOD  = 14

# ── Wave detection (spec v2, Steps 1-4) ───────────────────────────────────────
MIN_WAVE    = 3        # bars; a shorter wave is merged into its neighbours
WHIP_FRAC   = 0.05     # a cross counts once |MACD - signal| reaches this share...
WHIP_RANGE  = 50       # ...of the MACD range over this many bars...
WHIP_BARS   = 3        # ...within this many bars, the cross bar being the first
WHIP_ATR    = 0.05     # ...and also reaches this many ATR (a tiny range is not a real gap)
SPACING     = (8, 80)  # bars allowed between MACD trough A and MACD trough B
BUMP_MIN    = 0.25     # height of the middle of the W, as a share of the leg's MACD range
PRICE_TAIL  = 2        # a wave's price extreme may print up to 2 bars after the wave ends
MAX_BUMPS   = 1        # 2 = the spec's extended mode (off by default)
STOP_ATR       = 0.25   # the stop sits this many ATR beyond price extreme B
OI_AGAINST_PCT = 5.0    # OI rising more than this into B, against the signal, costs points
GRADE_A, GRADE_B = 70, 55

# Per-timeframe settings. `fresh_h` is how many hours after its close a signal
# bar stays enterable (counted in bars, so the backtest stays deterministic).
# htf / ltf are the timeframes used for agreement and for the early-entry trigger.
TF_CFG = {
    "4h": {"label": "4H", "minutes": 240, "bars": 720, "fresh_h": 24,
           "htf_minutes": 1440, "ltf": "1h", "ltf_minutes": 60},
    "1h": {"label": "1H", "minutes": 60,  "bars": 500, "fresh_h": 8,
           "htf_minutes": 240, "ltf": "15m", "ltf_minutes": 15},
}
LTF15_WINDOW_H = 8    # a 4H signal's 15m trigger must close within this many hours of the 4H close

# Minimum divergence size. Strict also wants both troughs below zero
# (both peaks above zero for bearish).
STRICTNESS = {
    "balanced": {"min_price_pct": 0.25, "atr_frac": 0.20, "min_macd_frac": 0.15,
                 "zero_line": False},
    "strict":   {"min_price_pct": 0.50, "atr_frac": 0.55, "min_macd_frac": 0.22,
                 "zero_line": True},
}

ALLOWED_EXCHANGES = {"okx", "binance", "bybit"}
DEFAULT_TYPE      = {"okx": "swap", "binance": "future", "bybit": "swap"}
# exchange=all: Binance is the primary venue (you trade there); a coin it does not
# list is scanned on whichever of OKX / Bybit trades it more. Tags name venues in
# VENUE_ORDER.
PRIMARY_VENUE = "binance"
VENUE_ORDER   = ("okx", "bybit", "binance")

# Thin-coin filters (skipped under a symbol override, like the volume gate).
# Unknown data (no bid/ask, no OI) never filters.
MAX_SPREAD_PCT = 0.10            # top-of-book spread at scan time
ATR_PCT_RANGE  = (0.6, 8.0)      # 4H ATR as % of price: below is too dead for 1R, above too wild
MIN_HISTORY    = 300             # completed bars a timeframe needs (the 200 EMA has settled)
MIN_OI_USD     = 5_000_000       # open interest of a pair with a signal
STABLE_BASES      = {"USDC","BUSD","TUSD","USDP","DAI","FDUSD","USDD","USDE",
                     "EUR","USD1","PYUSD","XAUT","PAXG","EURC","USDG"}

# ── TradFi perp filter ─────────────────────────────────────────────────────────
# All three exchanges now list stock / ETF / commodity perps as ordinary linear
# USDT swaps (Bybit ~170, OKX ~160 + 8 commodities, Binance ~180 — TSLA, NVDA,
# BABA, SONY, SPY, oil, gold, ...). Tesla clears any sane volume floor, so
# without a filter they land in the scan next to the coins — on an instrument
# tracking a market that CLOSES, which the divergence logic was never meant for.
#
# Primary check is each exchange's own instrument metadata (see is_tradfi) —
# that is what separates METAUSDT-the-stock from META-the-crypto-token and
# needs no upkeep as they list more. The list below is only a name backstop in
# case a venue ships an untagged listing. Curated, NOT a pattern: plain
# META/OPEN/MAX are deliberately absent because those are also real crypto
# tokens; the metadata check catches the stock versions where they exist.
EQUITY_BASES      = {
    # plain equity tickers
    # (no plain META/OPEN/MAX — those tickers are real crypto tokens)
    "TSLA","NVDA","AAPL","GOOGL","GOOG","AMZN","MSFT","COIN","HOOD","MSTR",
    "CRCL","SPY","QQQ","GME","PLTR","AMD","INTC","NFLX","ORCL","AVGO",
    # xStock (Backed) variants
    "TSLAX","NVDAX","AAPLX","GOOGLX","AMZNX","MSFTX","METAX","COINX","HOODX",
    "MSTRX","CRCLX","SPYX","QQQX","GMEX","PLTRX","AMDX","INTCX","NFLXX",
    "ORCLX","AVGOX","OPENX","LLYX","UNHX","VX","JNJX","PGX","KOX","WMTX",
    "XOMX","CVXX","MCDX","CSCOX","IBMX","BAX","PFEX","ABTX","MRKX","TMOX",
    "ACNX","LINX","DHRX","NVOX","AZNX","GLDX","TQQQX","ABBVX","BRKBX",
    "CRWDX","DFDVX","HONX","MDTX","MRVLX","NKEX","PMX","BACX","JPMX",
    "CRMX","PEPX","HDX","APPX","AMBRX","TBLLX",
}


def is_tradfi(mkt):
    """True for a stock / ETF / commodity perp, from the exchange's own
    instrument metadata (verified against live listings, Aug 2026):

      bybit    info.symbolType   in stock / ETF / commodity
                                                (crypto: "" / "innovation")
      binance  info.contractType == "TRADIFI_PERPETUAL"  (crypto: "PERPETUAL")
      okx      info.instCategory == "3" equities/pre-IPO, "4" commodities
                                                (crypto: "1")

    Falls back to the EQUITY_BASES name list for anything untagged.
    """
    mkt = mkt or {}
    info = mkt.get("info") or {}
    if str(info.get("symbolType", "")).lower() in ("stock", "etf", "commodity"):
        return True
    if str(info.get("contractType", "")).upper().startswith("TRADIFI"):
        return True
    if str(info.get("instCategory", "")) in ("3", "4"):
        return True
    return mkt.get("base") in EQUITY_BASES

DEFAULT_MIN_WEEKLY_VOL = 70_000_000      # $70M / 7 days ≈ $10M/day
DEFAULT_MAX_PAIRS      = 400
PREGATE_SLACK          = 0.30            # loose: the real gate is the 7-day measurement
MOVED_PCT              = 2.0             # beyond this the setup is "extended"
DISPLAY_TZ_OFFSET_H    = 8               # PHT / UTC+8 for displayed timestamps

# Threading: workers and requests/sec per exchange. OKX's candle endpoints are
# the tightest of the three, so it gets its own (lower) numbers.
SPEED = {
    "safe":   {"workers": 3,  "rps": {"okx": 4,  "binance": 6,  "bybit": 6}},
    "normal": {"workers": 6,  "rps": {"okx": 8,  "binance": 14, "bybit": 14}},
    "fast":   {"workers": 10, "rps": {"okx": 14, "binance": 22, "bybit": 22}},
}

# ── OI flow (context tags) ─────────────────────────────────────────────────────
# The "check OI on Sunday" routine, mechanised. These thresholds classify OI flow
# for a coin that already has a divergence: whether its open interest moved
# meaningfully over the lookback; direction of price and
# OI together give the quadrant, and funding says whether the move is spot-led
# or leveraged-crowded. Thresholds are heuristics, stated here once:
OI_LOOKBACK_D      = 7      # compare OI now vs ~7 days ago
OI_MIN_CHANGE_PCT  = 5.0    # below this the pair's OI flow is "quiet" and gets no tag
OI_MIN_HISTORY     = 6      # fewer OI points than this = fresh listing → skip.
                            # A new listing's OI always ramps from zero; that is
                            # the classic false positive of this method.
# Funding is normalised to % PER DAY (intervals differ per contract: 1h/4h/8h).
# Neutral perp funding is 0.01%/8h ≈ 0.03%/day.
FUND_SPOT_LED_MAX  = 0.02   # ≤ this with price+OI up: buyers are in spot, not leverage
FUND_CROWDED_MIN   = 0.10   # ≥ this with price+OI up: longs paying 3x+ neutral — late & crowded
FUND_SQUEEZE_MAX   = -0.10  # ≤ this with price down + OI up: shorts crowded — squeeze fuel
OI_VOL_RATIO_HIGH  = 1.5    # OI larger than 1.5x average daily volume = crowded either way

# ── Daily movers ───────────────────────────────────────────────────────────────
# A different question from everything above: no MACD, no divergence, no OI.
# "Which liquid perps did something UNUSUAL in the last completed day, and what
# level is that move sitting against?" — i.e. what a daily watchlist post is
# actually built from, made explicit.
#
# Three triggers; any ONE qualifies a pair:
#   move    |1-day close-to-close change|  >= MOVER_MIN_CHG_PCT
#   volume  last day's traded value        >= MOVER_VOL_MULT  x its 20-day average
#   range   last day's true range          >= MOVER_RANGE_MULT x ATR20
#
# The volume and range tests are the ones that earn their keep. A coin can be
# up 5% on nothing; 3x its normal volume with a range twice its ATR is someone
# actually doing something. Sorting by raw % change — the lazy version of this
# scan — systematically misses dormant coins waking up, which is exactly the
# kind of name that makes these lists interesting.
#
# Everything is measured on COMPLETED daily bars, so a scan at 02:00 and one at
# 22:00 agree about the same day. That means the scan describes YESTERDAY and
# you trade it TODAY, which is the honest version of a "daily watchlist".
MOVER_LOOKBACK_D  = 20      # baseline window for "normal" volume and range
MOVER_RANGE_D     = 7       # the range whose boundaries get quoted as levels
MOVER_MIN_CHG_PCT = 5.0
MOVER_VOL_MULT    = 2.0
MOVER_RANGE_MULT  = 1.5
MOVER_EDGE_PCT    = 2.0     # within this % of a 7-day boundary = "at the edge"

# Thin band (opt-in, /movers?thin=1). Pairs between this and the volume floor
# are measured too, but listed separately and held to a stricter test: volume
# spike AND range expansion together. On a $3M/day perp one whale order is a
# "7x volume day" on its own; needing the range to expand with it is what
# separates real participation from a single fill. The costs that justify the
# main floor (spread, slippage through stops, wicks that clear stops and snap
# back) do not go away — the band is labelled so size can come down with it.
MOVER_THIN_MIN_WEEKLY_VOL = 20_000_000   # ≈ $3M/day


STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scan_state.json")
# Own file on purpose: a movers scan must never be able to clobber the
# divergence scan's saved state, and vice versa.
MOVERS_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "movers_state.json")
STATE_VERSION = 3    # scan_state.json written by an older row shape is ignored
HTF_BARS_1D   = 200   # daily history, for the 4H signal's higher-timeframe check
LTF_BARS_15M  = 300   # 15m history (75h), for the 1H and 4H signals' early-entry triggers

# ── News calendar ──────────────────────────────────────────────────────────────
# Scheduled macro prints, so a setup is never sized into a CPI or FOMC release
# by accident. JBlanked (ForexFactory's data behind a free API key) is the
# primary source; ForexFactory's own weekly JSON is the keyless fallback.
CAL_STATE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "calendar_state.json")
CAL_REFRESH_MIN   = 30    # minutes between ForexFactory fetches (free feed)
# JBlanked's calendar endpoints are metered: every call costs an account credit.
# It only supplies the schedule past the ForexFactory week, which barely moves,
# so it runs on a much slower clock than the feed carrying today's forecasts —
# 4 credits a day instead of 48.
CAL_JB_REFRESH_MIN = 360
CAL_MIN_REFRESH_S = 300   # floor for a manual Refresh, so a click cannot spend credits in a loop
CAL_DAYS_AHEAD    = 14    # JBlanked range horizon (ForexFactory only serves this week)
JB_KEY_ENV        = "JBLANKED_API_KEY"   # read from the environment or a .env file
JB_CAL_URL        = "https://www.jblanked.com/news/api/forex-factory/calendar/range/"
# JBlanked serves broker time and its `offset` parameter is loosely documented.
# Whatever it returns, the clock difference is measured against the shared
# ForexFactory events on every refresh and corrected (merge_calendars); this
# constant only sets the request so the correction is as small as possible.
JB_TIME_OFFSET    = 3
FF_CAL_URL        = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# "Market movers": the prints that actually move crypto perps. Everything else
# ForexFactory rates High (RBA speeches, Australian jobs, weekly claims, ...) is
# hidden by default. Each rule is a currency plus a regex on the event title.
CAL_MOVERS = [
    ("USD", r"^(core )?cpi\b"),                     # CPI m/m, Core CPI m/m, CPI y/y
    ("USD", r"^core pce"),
    ("USD", r"^(core )?ppi\b"),
    ("USD", r"^non-farm employment"),
    ("USD", r"^unemployment rate"),                 # not the weekly Unemployment Claims
    ("USD", r"^average hourly earnings"),
    ("USD", r"^federal funds rate"),
    ("USD", r"^fomc (statement|press conference|meeting minutes|economic projections)"),
    ("USD", r"^fed chair \w+ (speaks|testifies)"),
    ("USD", r"^(advance|prelim|final) gdp"),
    ("USD", r"^(core )?retail sales"),
    ("USD", r"^ism (manufacturing|services) pmi"),
    ("JPY", r"^boj policy rate"),
    ("EUR", r"^main refinancing rate"),
    ("GBP", r"^official bank rate"),
]
_CAL_MOVERS_RE = [(c, re.compile(p, re.I)) for c, p in CAL_MOVERS]
# The USD movers that hold an entry (spec "Good to enter?" rule 10). Every
# other mover above only puts a MACRO tag on the row.
MACRO_BLOCKERS = [re.compile(p, re.I) for p in (
    r"^(core )?cpi\b", r"^core pce", r"^(core )?ppi\b", r"^non-farm employment",
    r"^unemployment rate", r"^federal funds rate", r"^fomc (statement|press conference)",
    r"^fed chair \w+ (speaks|testifies)")]

# ── Server state ───────────────────────────────────────────────────────────────
_state = {"results": [], "ts": None, "meta": {}}
_lock  = threading.Lock()
_movers_state = {"rows": [], "ts": None, "meta": {}}
_cal_state = {"events": [], "ts": None, "attempt": None, "meta": {},
              "jb": {"events": [], "ts": None, "attempt": None}}

# Candles and geometry of the last scan's signals, for /chart.
# shortcut: memory only, so charts need a fresh scan after a restart; persist
# a slice per signal in scan_state.json if that becomes annoying.
_charts = {}


# ── Indicators ─────────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def compute_indicators(df):
    """MACD line / signal / histogram, ATR14 (Wilder), MACD in ATR units and EMA 200,
    on completed bars."""
    close = df["close"]
    ml = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    sl = ema(ml, MACD_SIG)

    prev = close.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

    return pd.DataFrame({"macd": ml, "signal": sl, "hist": ml - sl, "atr": atr,
                         # zero ATR (flat opening bars) gives 0, not NaN, so dropna keeps the row
                         "macd_atr": (ml / atr.where(atr > 0)).fillna(0.0),
                         "ema200": ema(close, EMA_TREND)}, index=df.index)


def rsi(close, p=RSI_PERIOD):
    """Wilder RSI. Used on BTC 4H only, for the regime tag."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / p, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# ── Pivots ─────────────────────────────────────────────────────────────────────
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


# ── Wave divergence (spec v2) ─────────────────────────────────────────────────
def macd_crosses(macd, signal, atr=None):
    """Confirmed MACD/signal crossovers as [(bar, +1 bullish | -1 bearish)].

    The list alternates direction, so consecutive entries bound one wave. A
    cross that never opens a real gap (the whipsaw filter: WHIP_FRAC of the
    MACD range AND, when `atr` is given and positive, WHIP_ATR x ATR) is dropped, and a
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
        gap = (s * d[i:i + WHIP_BARS]).max()
        if gap < WHIP_FRAC * rng[i] or (atr is not None and atr[i] > 0 and gap < WHIP_ATR * atr[i]):
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
    crosses = macd_crosses(df["macd"].values, df["signal"].values, atr)

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



def px(v):
    """Round a price to a sensible number of places for its magnitude.

    round(x, 8) on a $73 coin prints 73.11755740 — nine characters of noise
    for a level you have to type into an exchange by hand. Precision scales
    with the price instead, so sub-cent tokens keep their digits and
    four-figure coins do not carry eight of them.
    """
    a = abs(v)
    d = 2 if a >= 1000 else 3 if a >= 100 else 4 if a >= 1 else 6 if a >= 0.01 else 8
    return round(v, d)




# ── Build a signal record ──────────────────────────────────────────────────────
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
        "bars_ago": n - 1 - sb, "fresh": (n - 1 - sb) * cfg["minutes"] / 60 <= cfg["fresh_h"],
        "forming": forming,
        "b_start_ms": ms(d["b_start"]), "b_ms": ms(bp), "signal_ms": ms(sb),
        "signal_close_ms": ms(sb) + cfg["minutes"] * 60_000,
        "above_ema200": bool(sp > float(df["ema200"].iloc[sb])),
        "price_atr": round(d["price_atr"], 3), "macd_frac": round(d["macd_frac"], 4),
        "bump_frac": round(d["bump_frac"], 4),
        "zero_ok": d["zero_ok"], "triple": d["triple"], "vol_dry": d["vol_dry"],
        "wick": d["wick"], "zero_cross": zero_cross,
        "_b": sb, "_d": d,
    }


# ── Status: what has price done since the pivot (no RSI involved) ─────────────
def evaluate_status(df, sig, live_price):
    """Mechanical, backward-looking state of the setup.

      invalidated  price traded through the stop → the swing broke, it's dead
      ran          price already reached 1R or better (r_reached says how far)
      extended     moved >2% in the signal direction but not yet 1R
      active       still sitting near the signal

    All of it is measured against the ORIGINAL signal-anchored stop and risk,
    so 'ran 2R' means the trade you'd have taken at the signal made 2R — not
    something relative to a re-anchored entry.

    Completed bars decide invalidation (so the verdict never repaints), with
    one exception: if the LIVE price is already through the stop, the setup is
    dead now and saying otherwise would be pedantry.
    """
    b    = sig["_b"]
    side = sig["side"]
    sp   = sig["signal_price"]
    stop = sig["stop_price"]
    risk = sig["signal_risk"]
    after = df.iloc[b + 1:]

    raw_pct = (live_price - sp) / sp * 100          # what price literally did
    pct = raw_pct * (1 if side == "bullish" else -1)  # ...in the signal's favour
    base = {"pct_chg": round(pct, 2), "price_chg_pct": round(raw_pct, 2)}
    dead = (live_price <= stop) if side == "bullish" else (live_price >= stop)

    if len(after) == 0:
        return dict(base, status="invalidated" if dead else "active", r_reached=0.0)

    if side == "bullish":
        if dead or float(after["low"].min()) <= stop:
            return dict(base, status="invalidated", r_reached=0.0)
        mfe = max(float(after["high"].max()), live_price) - sp
    else:
        if dead or float(after["high"].max()) >= stop:
            return dict(base, status="invalidated", r_reached=0.0)
        mfe = sp - min(float(after["low"].min()), live_price)

    r = max(0.0, mfe / risk) if risk > 0 else 0.0

    if r >= 1.0:
        status = "ran"
    elif pct >= MOVED_PCT:
        status = "extended"
    else:
        status = "active"

    return dict(base, status=status, r_reached=round(r, 2))


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
def score_signal(sig, htf_ok=None, oi_chg=None, fund=None, btc_state=None):
    """(0-100 score, grade, partial). A description of the setup, not a win rate.

    Pattern points 0-70, context 10 each: higher timeframe agrees, OI falling
    into B with funding against the crowd, BTC state on the signal's side.
    OI rising more than OI_AGAINST_PCT into B with funding on the crowded side
    costs 10. A context input that is None (htf_ok, oi_chg or fund, btc_state)
    is unavailable: that component leaves the denominator and `partial` is True.
    """
    bull = sig["side"] == "bullish"
    s = min(sig["price_atr"], 1.0) * 12                                   # 0-12
    s += min(sig["macd_frac"] / 0.5, 1.0) * 12                            # 0-12
    s += float(np.clip((sig["bump_frac"] - BUMP_MIN) / 0.5, 0, 1)) * 8    # 0-8
    s += 10 * sig["zero_ok"] + 10 * sig["vol_dry"] + 10 * sig["wick"] + 8 * sig["triple"]
    avail = 70
    if htf_ok is not None:
        avail += 10
        s += 10 * bool(htf_ok)
    if oi_chg is not None and fund is not None:
        avail += 10
        f = fund if bull else -fund
        if oi_chg < 0 and f <= 0:
            s += 10
        elif oi_chg > OI_AGAINST_PCT and f > 0:
            s -= 10
    if btc_state is not None:
        avail += 10
        s += 10 * (btc_state == ("BULL" if bull else "BEAR"))
    score = int(max(0, min(100, round(s / avail * 100))))
    return score, ("A" if score >= GRADE_A else "B" if score >= GRADE_B else "C"), avail < 100


def btc_state(rsi, close, ema200):
    """BTC 4H: BULL (RSI > 55 and close above the 200 EMA), BEAR (RSI < 45 and
    close below it), else CHOP. None when any input is missing."""
    if any(pd.isna(v) for v in (rsi, close, ema200)):
        return None
    if rsi > 55 and close > ema200:
        return "BULL"
    if rsi < 45 and close < ema200:
        return "BEAR"
    return "CHOP"


def regime_tag(side, above_ema200, state):
    """CHOP when BTC is CHOP; WITH TREND when the BTC state and the coin's 200 EMA
    both agree with the side; else (BTC unknown included) COUNTER TREND."""
    if state == "CHOP":
        return "CHOP"
    bull = side == "bullish"
    ok = state == ("BULL" if bull else "BEAR") and above_ema200 == bull
    return "WITH TREND" if ok else "COUNTER TREND"


def finish_signal(sig, htf_ok=None, oi_chg=None, fund=None, btc_state=None, trigger=None,
                  trigger15=None):
    """Score, grade, regime, trigger, tags and targets.

    Called once without context so a signal is always complete, and again by
    add_context() once the scan has fetched the outside data. `trigger` (the
    lower-timeframe trigger) and `trigger15` (a 4H signal's 15m trigger) are
    (price, close time in ms of the trigger bar) or None; the price comes from
    whichever closed first.
    """
    sig["score"], sig["grade"], sig["partial"] = score_signal(sig, htf_ok, oi_chg, fund, btc_state)
    sig["htf_ok"] = htf_ok
    sig["oi_chg_pct"] = None if oi_chg is None else round(oi_chg, 1)
    sig["btc_state"] = btc_state
    sig["regime"] = regime_tag(sig["side"], sig["above_ema200"], btc_state)
    first = min((t for t in (trigger, trigger15) if t), key=lambda t: t[1], default=None)
    sig["trigger_price"] = px(first[0]) if first else None
    # "first" = the trigger bar closed before the signal bar did
    sig["trigger_first"] = bool(first and (sig["forming"] or first[1] < sig["signal_close_ms"]))
    sig["tags"] = [t for t, on in (("TRIPLE", sig["triple"]),
                                   ("W-PATTERN", sig["bump_frac"] >= 0.5),
                                   ("LTF TRIGGER", bool(trigger)),
                                   ("15M TRIGGER", bool(trigger15)),
                                   ("ZERO-LINE CROSS", sig["zero_cross"]),
                                   ("PARTIAL DATA", sig["partial"])) if on]
    sig.update(build_targets(sig))
    return sig


# ── Outside context for a signal ───────────────────────────────────────────────
def asof(series, ts):
    """Last value at or before ts, or None."""
    s = series.loc[:ts]
    return float(s.iloc[-1]) if len(s) else None


def htf_agrees(htf, close_ts, htf_minutes, side):
    """Was the higher-timeframe MACD line rising (falling for bearish) on its
    last bar completed by close_ts? Frames are indexed by bar OPEN time, so a
    bar is complete once open + htf_minutes <= close_ts. None (unavailable)
    without a frame or with fewer than 2 completed bars."""
    if htf is None:
        return None
    m = htf["macd"].loc[:close_ts - pd.Timedelta(minutes=htf_minutes)]
    if len(m) < 2:
        return None
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
    # shortcut: nearest point at or before each time, so a history coarser than
    # the wave gives None; grade_signals then tries the daily one after the hourly.
    pts = sorted((p for p in (hist or []) if p.get("timestamp")), key=lambda p: p["timestamp"])
    for key in ("openInterestAmount", "openInterestValue"):
        a = [p for p in pts if p["timestamp"] <= t0_ms and p.get(key)]
        b = [p for p in pts if p["timestamp"] <= t1_ms and p.get(key)]
        if a and b and b[-1] is not a[-1]:
            return (float(b[-1][key]) / float(a[-1][key]) - 1) * 100.0
    return None


def btc_state_at(btc, close_ts):
    """BTC state on the last 4H bar completed by close_ts (bars indexed by open
    time), from the fetch_btc() frame. None when there is no frame."""
    if btc is None:
        return None
    t = close_ts - pd.Timedelta(minutes=240)
    return btc_state(*(asof(btc[k], t) for k in ("rsi", "close", "ema200")))


def add_context(sig, htf, ltf, oi_hist, fund, btc, ltf15=None):
    """Re-grade a signal with everything fetched for it. Any argument may be
    None; that component is then unavailable and leaves the score's denominator.
    `ltf15` is the 15m frame for a 4H signal's second trigger (spec Step 8); it
    counts only if it closes within LTF15_WINDOW_H of the 4H close."""
    cfg = TF_CFG[sig["tf"]]
    close_ts = pd.Timestamp(sig["signal_close_ms"], unit="ms")
    since, start = pd.Timestamp(sig["b_ms"], unit="ms"), pd.Timestamp(sig["signal_ms"], unit="ms")
    trig = trig15 = None
    if ltf is not None and sig["fresh"]:
        trig = ltf_trigger(ltf, sig["side"], since, start, cfg["ltf_minutes"])
    if ltf15 is not None and sig["fresh"]:
        trig15 = ltf_trigger(ltf15, sig["side"], since, start, 15)
        if trig15 and trig15[1] > sig["signal_close_ms"] + LTF15_WINDOW_H * 3_600_000:
            trig15 = None
    return finish_signal(sig, htf_agrees(htf, close_ts, cfg["htf_minutes"], sig["side"]),
                         oi_change_pct(oi_hist, sig["b_start_ms"], sig["b_ms"]),
                         fund, btc_state_at(btc, close_ts), trig, trig15)


def grade_signals(sigs, df4, df1, d1, df15, oi_d, oi_h, fund, btc):
    """Give each signal the context that belongs to its timeframe.

    4H signal: higher timeframe = 1D (d1), lower = 1H (df1), plus the 15m trigger (df15).
    1H signal: higher timeframe = 4H (df4), lower = 15m (df15).
    Open interest: hourly history when it covers wave B, else daily.
    Any frame may be None.
    """
    for s in sigs:
        four = s["tf"] == "4h"
        hist = oi_h if oi_change_pct(oi_h, s["b_start_ms"], s["b_ms"]) is not None else oi_d
        add_context(s, d1 if four else df4, df1 if four else df15, hist, fund, btc,
                    df15 if four else None)
    return sigs


# ── Entry verdict ──────────────────────────────────────────────────────────────
MACRO_WAIT_H = {"1h": 4, "4h": 12}   # a USD blocker due inside this many hours means wait


def _hours_txt(h):
    return f"{h:.0f}h" if h >= 1 else f"{h * 60:.0f}m"


def entry_verdict(sig, blocker=None):
    """Is this setup enterable as of the scan? A CHECKLIST, not a prediction.

    First matching rule wins (spec "Good to enter?"):
      skip   stop hit · already ran 1R+ · signal bar too old · grade C
      wait   counter trend below grade A · price ran >MOVED_PCT from the signal ·
             a USD blocker due inside MACRO_WAIT_H or printed in the last
             hour · signal bar still forming with no lower-timeframe trigger
      enter  none of the above: "Yes (early)" on a forming signal bar,
             "Yes (confirmed)" on a closed one

    `blocker` is (hours until it prints, negative once printed, title) from
    macro_window(), or None. `level` is a price the dashboard prints after the
    reason, so it is formatted in one place (sub-cent prices need plain digits).
    """
    cfg = TF_CFG[sig["tf"]]
    def out(verdict, reason, level=None):
        return {"verdict": verdict, "reason": reason, "level": level}

    st = sig["status"]
    if st == "invalidated":
        return out("skip", "No. Price already hit the stop, so this divergence failed.")
    if st == "ran":
        return out("skip", f"No. Price already moved {sig['r_reached']:g}x the risk toward the "
                           "target. The move has happened; entering now is chasing it.")
    if not sig["fresh"]:
        return out("skip", f"No. {sig['bars_ago']} completed {cfg['label']} bars "
                           f"({sig['bars_ago'] * cfg['minutes'] / 60:g}h) have closed after the signal "
                           f"bar, too many to act on (the limit on {cfg['label']} is "
                           f"{cfg['fresh_h'] * 60 // cfg['minutes']} bars, {cfg['fresh_h']}h).")
    if sig["grade"] == "C":
        return out("skip", f"No. Weak setup: grade C, score {sig['score']} of 100.")
    if sig["regime"] == "COUNTER TREND" and sig["grade"] != "A":
        return out("wait", "Not yet. Watch only: this is against the trend, and a counter-trend "
                           f"setup needs grade A. This one is grade {sig['grade']}.")
    if st == "extended":
        return out("wait", f"Not yet. Price has already moved {sig['pct_chg']:g}% away from the "
                           "signal. Wait for it to pull back to",
                   sig["signal_price"])
    if blocker and -1 <= blocker[0] <= MACRO_WAIT_H[sig["tf"]]:
        h, title = blocker
        if h > 0:
            return out("wait", f"Not yet. {title} is due in {_hours_txt(h)} and can whip price. "
                               "Wait until it has passed.")
        return out("wait", f"Not yet. {title} printed {_hours_txt(-h)} ago. Give price the "
                           "first hour to settle.")
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


def macro_window(events, now):
    """(blocker, other), each (hours from now, title) or None.

    blocker: the nearest USD MACRO_BLOCKERS print, including one that printed
    within the last hour (negative hours). other: the nearest remaining curated
    mover due within 24h, which only tags the row."""
    blocker = other = None
    for e in events:
        ts = _cal_parse_ts("iso", e.get("ts"))
        if ts is None:
            continue
        h, title = (ts - now).total_seconds() / 3600.0, e.get("title", "")
        if e.get("currency") == "USD" and any(rx.search(title.strip()) for rx in MACRO_BLOCKERS):
            if h >= -1 and (blocker is None or h < blocker[0]):
                blocker = (h, title)
        elif e.get("mover") and 0 < h <= 24 and (other is None or h < other[0]):
            other = (h, title)
    return blocker, other


def brief_meta(btc, window, now):
    """Scan-meta fields for the page's brief.

    btc: the last completed BTC 4H bar of the fetch_btc() frame as {state, rsi,
    close, ema200}, or None without a frame. blocker / macro: the two halves of
    macro_window() (`window`, taken at `now`) as {ts_ms, title} or None; the
    time is absolute so the page can count down live. `window` None (calendar
    unavailable) leaves both keys out, so the page says nothing about macro."""
    out = {"btc": None}
    if btc is not None and len(btc):
        last = btc.iloc[-1]
        out["btc"] = {"state": btc_state(last["rsi"], last["close"], last["ema200"]),
                      "rsi": round(float(last["rsi"]), 1),
                      "close": px(float(last["close"])), "ema200": px(float(last["ema200"]))}
    if window is not None:
        for key, e in zip(("blocker", "macro"), window):
            out[key] = e and {"ts_ms": int((now.timestamp() + e[0] * 3600) * 1000), "title": e[1]}
    return out


# ── Exchange helpers ───────────────────────────────────────────────────────────
def make_exchange(exid, rate_limit=True):
    """Build a ccxt exchange from an allowlisted id — never getattr on raw user
    input (v2 would instantiate any attribute of the ccxt module you passed in
    the query string)."""
    if exid not in ALLOWED_EXCHANGES:
        raise ValueError(f"exchange must be one of: {', '.join(sorted(ALLOWED_EXCHANGES))}")
    return getattr(ccxt, exid)({
        "enableRateLimit": rate_limit,
        "timeout": 20000,
        "options": {"defaultType": DEFAULT_TYPE[exid]},
    })


def usd_volume(ticker):
    """24h quote (USD) volume from a ccxt ticker, handling per-exchange quirks.

    THE $0 BUG (v2): for OKX derivatives ccxt hardcodes quoteVolume=None
    (okx.parse_ticker: `quoteVolume = ... if spot else None`), so
    `.get("quoteVolume") or 0` returned 0 for every swap, and sorting on
    all-zero keys is stable → the "volume rank" was just OKX listing order.

    For OKX linear swaps, info.volCcy24h is 24h volume in BASE currency →
    × last price = USDT volume. (ccxt's baseVolume for OKX swaps is in
    CONTRACTS, not coins — never multiply that.)

    Returns None when volume is genuinely unknown. Callers must treat None as
    "unknown", never as zero.
    """
    if not ticker:
        return None
    try:
        qv = ticker.get("quoteVolume")
        if qv:
            return float(qv)
        info = ticker.get("info") or {}
        tv = info.get("turnover24h")             # bybit: quote-ccy turnover
        if tv:
            return float(tv)
        last = ticker.get("last") or ticker.get("close")
        vc = info.get("volCcy24h")               # okx derivs: base-ccy amount
        if vc and last:
            return float(vc) * float(last)
    except (TypeError, ValueError):
        return None
    return None


def weekly_usd_volume(df, usd_24h, tf_minutes):
    """Rolling 7-day traded value in USD.

    The honest problem: ccxt's OHLCV `volume` column is in different units per
    exchange and per market type — base coins on some, CONTRACTS on OKX
    derivatives, quote currency on others. Multiplying by contractSize and
    price gets it right only if you guessed the unit correctly, and guessing
    wrong is off by orders of magnitude.

    So this never needs to know the unit. It takes the 24h USD volume from the
    ticker (which usd_volume() already normalises per exchange) and scales it
    by the RATIO of 7-day to 24h traded notional from the candles. Whatever
    constant factor the volume column carries divides straight out of the
    ratio. Falls back to 24h × 7 when there aren't enough candles.
    """
    if not usd_24h:
        return None
    bars_24h = max(1, int(1440 / tf_minutes))
    bars_7d  = bars_24h * 7
    if df is None or len(df) < bars_7d:
        return usd_24h * 7.0
    notional = (df["volume"] * df["close"]).astype(float)
    d = float(notional.iloc[-bars_24h:].sum())
    w = float(notional.iloc[-bars_7d:].sum())
    if d <= 0 or w <= 0:
        return usd_24h * 7.0
    return usd_24h * (w / d)


# ── OI flow helpers ────────────────────────────────────────────────────────────
def fetch_funding_map(ex):
    """symbol → funding rate normalised to % PER DAY, from ONE bulk call.

    Raw funding rates are per-interval, and intervals differ per contract
    (1h / 4h / 8h). A 0.01% hourly rate is 8x the cost of a 0.01% 8-hour rate,
    so comparing raw numbers across pairs is meaningless — everything is
    converted to a daily percentage first. Interval comes from ccxt's parsed
    'interval' where present, else the raw fundingInterval (minutes, bybit),
    else the classic 8h default.
    """
    try:
        rates = ex.fetch_funding_rates()
    except Exception:
        return {}
    out = {}
    for sym, r in rates.items():
        fr = r.get("fundingRate")
        if fr is None:
            continue
        hours = None
        iv = r.get("interval")
        if iv:
            try:
                hours = float(str(iv).lower().rstrip("h"))
            except (TypeError, ValueError):
                hours = None
        if not hours:
            try:
                hours = float((r.get("info") or {}).get("fundingInterval")) / 60.0
            except (TypeError, ValueError):
                hours = 0.0
        if not hours:
            hours = 8.0
        out[sym] = float(fr) * (24.0 / hours) * 100.0
    return out


def oi_point_usd(pt, fallback_price):
    """USD value of one OI history point. Venues disagree on units the same way
    they do for volume: okx reports value, bybit only the coin amount, binance
    both. Value wins when present; amount x price otherwise."""
    try:
        v = pt.get("openInterestValue")
        if v:
            return float(v)
        a = pt.get("openInterestAmount")
        if a and fallback_price:
            return float(a) * float(fallback_price)
    except (TypeError, ValueError):
        pass
    return None


def spread_too_wide(ticker):
    """Top-of-book spread above MAX_SPREAD_PCT of mid. Missing bid/ask: False."""
    try:
        bid, ask = float(ticker["bid"]), float(ticker["ask"])
    except (TypeError, KeyError, ValueError):
        return False
    if not (bid > 0 and ask > 0):
        return False
    return (ask - bid) / ((ask + bid) / 2) * 100 > MAX_SPREAD_PCT


def atr_pct_out_of_range(atr, close):
    """4H ATR as % of price outside ATR_PCT_RANGE. Unknown (not > 0): False."""
    if not (atr > 0 and close > 0):
        return False
    pct = atr / close * 100
    return not ATR_PCT_RANGE[0] <= pct <= ATR_PCT_RANGE[1]


def oi_too_thin(hist, price):
    """Latest point of an OI history below MIN_OI_USD. Unknown: False."""
    # shortcut: OKX's OI history is per coin (all its contracts), so there this is coin-wide, not this contract's; filter per contract from fetch_open_interest if a coin with several OKX contracts slips through.
    pts = [p for p in (hist or []) if p.get("timestamp")]
    usd = oi_point_usd(max(pts, key=lambda p: p["timestamp"]), price) if pts else None
    return usd is not None and usd < MIN_OI_USD


def coin_venues(perps, vols, primary):
    """Which venue each coin is scanned on.

    perps: {venue: [symbol]}; vols: {(venue, symbol): 24h USD volume or None}.
    Returns ({base: {venue: symbol}} for every venue listing the coin, and
    [(venue, symbol)], one per coin: the primary venue when it lists the coin,
    else whichever other venue has the larger 24h volume (OKX on a tie).
    shortcut: coins match by base name only (1000PEPE on one venue and PEPE on
    another count as two coins); add an alias map if that matters.
    """
    listed = {}
    for v, syms in perps.items():
        for s in syms:
            listed.setdefault(s.split("/")[0], {})[v] = s
    pick = []
    for by in listed.values():
        v = primary if primary in by else max(
            (x for x in VENUE_ORDER if x in by), key=lambda x: vols.get((x, by[x])) or 0)
        pick.append((v, by[v]))
    return listed, pick


def venue_tags(exid, confirmed):
    """Row tags of the all-exchanges scan: NOT ON BINANCE for a coin scanned
    elsewhere, ALSO ON ... naming the venues whose detection confirmed it."""
    tags = [] if exid == PRIMARY_VENUE else [f"NOT ON {PRIMARY_VENUE.upper()}"]
    if confirmed:
        tags.append("ALSO ON " + ", ".join(v.upper() for v in VENUE_ORDER if v in confirmed))
    return tags


def fetch_oi_history(ex, limiter, sym, limit, tf="1d", tries=3):
    """Daily OI history with the same backoff discipline as fetch_ohlcv."""
    for attempt in range(tries):
        try:
            limiter.acquire()
            return ex.fetch_open_interest_history(sym, timeframe=tf, limit=limit)
        except Exception as e:
            msg = str(e)
            if any(t in msg for t in _TRANSIENT) and attempt < tries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None


def classify_flow(price_chg, oi_chg, fund_day):
    """Quadrant + health flags — the 'good vs bad' filter.

    Quadrant is just the two signs. The flags are where the judgement lives:
    rising price + rising OI is only the textbook-healthy case when longs are
    NOT paying through the nose for it (spot-led); the same quadrant with rich
    positive funding is latecomer leverage. Price down + OI up + deeply
    negative funding means shorts are crowded and paying — squeeze fuel.
    """
    if oi_chg >= 0:
        quad = "NEW LONGS" if price_chg >= 0 else "NEW SHORTS"
    else:
        quad = "SHORT COVERING" if price_chg >= 0 else "CAPITULATION"
    flags = []
    if fund_day is not None:
        if quad == "NEW LONGS":
            if fund_day <= FUND_SPOT_LED_MAX:
                flags.append("SPOT-LED")
            elif fund_day >= FUND_CROWDED_MIN:
                flags.append("CROWDED LONGS")
        elif quad == "NEW SHORTS" and fund_day <= FUND_SQUEEZE_MAX:
            flags.append("SQUEEZE FUEL")
    return quad, flags


def flow_context(hist, df, live_price, fund_day, wk_vol):
    """OI-flow read for one pair from its daily OI history and daily candles.

    None means "nothing to say": too little history (a fresh listing's OI
    always ramps from zero) or no baseline point near the lookback. OI that
    moved less than OI_MIN_CHANGE_PCT comes back with the change but no tags.
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
    if abs(oi_chg) < OI_MIN_CHANGE_PCT:    # quiet: a number for the 7d OI column, no tags
        return {"quadrant": None, "flags": [], "oi_chg_pct": round(oi_chg, 1)}
    span = min(OI_LOOKBACK_D + 1, len(df))
    price_chg = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-span]) - 1) * 100.0

    quad, flags = classify_flow(price_chg, oi_chg, fund_day)
    if wk_vol and oi_now / (wk_vol / 7.0) >= OI_VOL_RATIO_HIGH:
        flags.append("HIGH OI/VOL")
    return {"quadrant": quad, "flags": flags,
            "oi_chg_pct": round(oi_chg, 1), "price_chg_pct": round(price_chg, 1),
            "funding_day_pct": round(fund_day, 4) if fund_day is not None else None}


# ── Daily movers helpers ───────────────────────────────────────────────────────
def analyse_mover(df, live_price):
    """Per-pair daily statistics, from COMPLETED bars only.

    Volume is used ONLY as a ratio against its own 20-day average. The OHLCV
    volume column carries an unknown constant factor per exchange and market
    type (base coins / contracts / quote currency), and that factor divides
    straight out of a self-ratio — the same trick weekly_usd_volume() uses, for
    the same reason. Never read vol_mult as a dollar amount or compare it
    across venues as one.
    """
    need = MOVER_LOOKBACK_D + 2
    if df is None or len(df) < need:
        return None

    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    prev_close, last_close = float(close[-2]), float(close[-1])
    chg_pct = (last_close / prev_close - 1) * 100 if prev_close else 0.0

    # True range per bar (needs the prior close, so the series starts at bar 1),
    # then the last completed day measured against the 20 bars before it.
    pc = close[:-1]
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - pc), np.abs(low[1:] - pc)))
    tr_last  = float(tr[-1])
    base_tr  = tr[-(MOVER_LOOKBACK_D + 1):-1]
    atr_base = float(base_tr.mean()) if len(base_tr) else 0.0
    range_mult = (tr_last / atr_base) if atr_base > 0 else None

    notional = (df["volume"] * df["close"]).astype(float).values
    base_vol = notional[-(MOVER_LOOKBACK_D + 1):-1]
    avg_vol  = float(base_vol.mean()) if len(base_vol) else 0.0
    vol_mult = (float(notional[-1]) / avg_vol) if avg_vol > 0 else None

    win  = df.iloc[-MOVER_RANGE_D:]
    r_hi = float(win["high"].max())
    r_lo = float(win["low"].min())
    span = r_hi - r_lo
    pos  = ((live_price - r_lo) / span * 100) if span > 0 else 50.0

    return {
        "chg_1d_pct":   round(chg_pct, 2),
        "chg_live_pct": round((live_price / last_close - 1) * 100, 2) if last_close else 0.0,
        "range_mult":   round(range_mult, 2) if range_mult is not None else None,
        "vol_mult":     round(vol_mult, 2) if vol_mult is not None else None,
        "pdh":  px(float(high[-1])), "pdl": px(float(low[-1])),
        "r_hi": px(r_hi),            "r_lo": px(r_lo),
        "pos_in_range": round(float(np.clip(pos, 0, 100)), 1),
        # Negative means price has already traded through that boundary.
        "to_r_hi_pct": round((r_hi - live_price) / live_price * 100, 2) if live_price else None,
        "to_r_lo_pct": round((live_price - r_lo) / live_price * 100, 2) if live_price else None,
    }


def classify_mover(m, thin=False):
    """(qualifies, flags, score) for one analysed pair.

    The score ranks how UNUSUAL the day was and how close price sits to a
    decision level. It carries NO direction and is NOT a win rate. It is a
    description, not a prediction.

    thin=True is the stricter test for the low-liquidity band: volume spike
    AND range expansion, where a liquid pair needs any one trigger.
    """
    vm  = m.get("vol_mult") or 0.0
    rm  = m.get("range_mult") or 0.0
    chg = abs(m.get("chg_1d_pct") or 0.0)
    hi_d, lo_d = m.get("to_r_hi_pct"), m.get("to_r_lo_pct")

    flags = []
    if vm >= MOVER_VOL_MULT:
        flags.append("VOLUME SPIKE")
    if rm >= MOVER_RANGE_MULT:
        flags.append("RANGE EXPANSION")
    if chg >= MOVER_MIN_CHG_PCT:
        flags.append("BIG MOVE")
    if hi_d is not None and hi_d <= 0:
        flags.append("ABOVE 7D HIGH")
    elif hi_d is not None and hi_d <= MOVER_EDGE_PCT:
        flags.append("AT RANGE HIGH")
    if lo_d is not None and lo_d <= 0:
        flags.append("BELOW 7D LOW")
    elif lo_d is not None and lo_d <= MOVER_EDGE_PCT:
        flags.append("AT RANGE LOW")

    if thin:
        qualifies = vm >= MOVER_VOL_MULT and rm >= MOVER_RANGE_MULT
    else:
        qualifies = (vm >= MOVER_VOL_MULT or rm >= MOVER_RANGE_MULT
                     or chg >= MOVER_MIN_CHG_PCT)

    # Distance to the NEARER 7-day boundary. Already through one → 0 → full marks.
    edge = min([d for d in (hi_d, lo_d) if d is not None] or [99.0])
    score  = float(np.clip((vm - 1.0) / 3.0, 0, 1)) * 35            # volume anomaly
    score += float(np.clip((rm - 0.8) / 1.7, 0, 1)) * 30            # range expansion
    score += float(np.clip(1.0 - max(edge, 0.0) / 6.0, 0, 1)) * 20  # near a level
    score += float(np.clip(chg / 12.0, 0, 1)) * 15                  # size of the move
    return qualifies, flags, int(max(0, min(100, round(score))))


def save_movers():
    try:
        with _lock:
            rows = list(_movers_state["rows"])
            ts   = _movers_state["ts"]
            meta = dict(_movers_state["meta"])
        with open(MOVERS_STATE_FILE, "w") as f:
            json.dump({"rows": rows, "ts": ts.isoformat() if ts else None,
                       "meta": meta}, f)
    except Exception as e:
        print(f"[movers] save failed: {e}")


def load_movers():
    try:
        with open(MOVERS_STATE_FILE) as f:
            data = json.load(f)
        ts = data.get("ts")
        with _lock:
            _movers_state["rows"] = data.get("rows", [])
            _movers_state["ts"]   = datetime.datetime.fromisoformat(ts) if ts else None
            _movers_state["meta"] = data.get("meta", {})
        print(f"[movers] restored {len(_movers_state['rows'])} row(s) from last scan")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[movers] load failed: {e}")


# ── News calendar ──────────────────────────────────────────────────────────────
def load_dotenv(path=None):
    """Minimal KEY=VALUE loader so the API key can live in a gitignored .env
    without adding python-dotenv. Keys are upper-cased; the real environment
    wins over the file. Returns how many variables were set."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return 0
    n = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().upper()
        if k.startswith("EXPORT "):
            k = k[7:].strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


def jb_api_key():
    return (os.environ.get(JB_KEY_ENV) or "").strip()


def is_mover(currency, title):
    t = (title or "").strip()
    return any(c == currency and rx.search(t) for c, rx in _CAL_MOVERS_RE)


def _cal_val(v, zero_is_missing=False):
    """Feed values as display strings: ForexFactory sends '0.3%' (or '' when
    unknown); JBlanked sends 0.3, and 0 wherever it has no value."""
    if v is None:
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if zero_is_missing and v == 0:
            return ""
        if float(v).is_integer():
            return str(int(v))
    return str(v).strip()


def _cal_parse_ts(source, raw):
    """JBlanked: '2024.02.08 15:30:00', already shifted to GMT by JB_TIME_OFFSET.
    ForexFactory / stored events: ISO-8601 with an offset or a trailing Z."""
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    try:
        if source == "jblanked":
            d = datetime.datetime.strptime(raw, "%Y.%m.%d %H:%M:%S")
            d = d.replace(tzinfo=datetime.timezone.utc)
        else:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            d = datetime.datetime.fromisoformat(raw)
            if d.tzinfo is None:
                d = d.replace(tzinfo=datetime.timezone.utc)
        return d.astimezone(datetime.timezone.utc)
    except ValueError:
        return None


def _iso_z(d):
    return d.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_events(rows, source):
    """Both feeds -> one shape: UTC ts, currency, impact, title, forecast,
    previous, actual, mover. Sorted by time, deduped, unusable rows dropped."""
    out, seen = [], set()
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        if source == "jblanked":
            title, cur, imp = r.get("Name"), r.get("Currency"), r.get("Impact")
            fc, pv, ac = r.get("Forecast"), r.get("Previous"), r.get("Actual")
            ts = _cal_parse_ts(source, r.get("Date"))
        else:
            title, cur, imp = r.get("title"), r.get("country"), r.get("impact")
            fc, pv, ac = r.get("forecast"), r.get("previous"), r.get("actual")
            ts = _cal_parse_ts(source, r.get("date"))
        if ts is None or not title:
            continue
        title, cur = str(title).strip(), str(cur or "").strip().upper()
        key = (ts, cur, title)
        if key in seen:
            continue
        seen.add(key)
        jb = source == "jblanked"
        out.append({"ts": _iso_z(ts), "currency": cur,
                    "impact": str(imp or "").strip() or "None", "title": title,
                    "forecast": _cal_val(fc, jb), "previous": _cal_val(pv, jb),
                    "actual": _cal_val(ac, jb), "mover": is_mover(cur, title)})
    out.sort(key=lambda e: (e["ts"], e["currency"], e["title"]))
    return out


def _pair_events(events, reference, shift_h=0.0):
    """Match events to reference entries one-to-one on currency + title.

    Closest pairs win first and each side is consumed once, so a title that
    repeats within a day (FOMC speakers, ECB speakers) pairs up in order
    instead of every occurrence collapsing onto whichever one happens to be
    nearest. `shift_h` is subtracted from the event times before measuring, so
    once the clock offset is known the pairing is done on corrected times.
    Returns [(event_index, reference_index, delta_hours), ...]."""
    by_key = {}
    for j, r in enumerate(reference):
        by_key.setdefault((r["currency"], r["title"]), []).append(j)
    ref_ts = [_cal_parse_ts("iso", r["ts"]) for r in reference]
    cands = []
    for i, e in enumerate(events):
        t = _cal_parse_ts("iso", e["ts"]) - datetime.timedelta(hours=shift_h)
        for j in by_key.get((e["currency"], e["title"]), []):
            d = (t - ref_ts[j]).total_seconds() / 3600.0
            if abs(d) <= 24:
                cands.append((abs(d), i, j, d + shift_h))
    cands.sort()
    used_e, used_r, pairs = set(), set(), []
    for _, i, j, d in cands:
        if i in used_e or j in used_r:
            continue
        used_e.add(i)
        used_r.add(j)
        pairs.append((i, j, d))
    return pairs


def calendar_time_check(events, reference):
    """Median hour difference between events present in both lists. None if
    nothing overlaps. Anything but ~0 means JBlanked is on a different clock,
    which merge_calendars then corrects."""
    deltas = [d for _, _, d in _pair_events(events, reference)]
    if not deltas:
        return None
    return round(statistics.median(deltas), 2)


def merge_calendars(ff_events, jb_events):
    """ForexFactory is authoritative for the week it serves: it keeps every
    event JBlanked drops (about a third, High-impact ones included), its times
    carry real UTC offsets and its numbers have units. JBlanked only extends
    the horizon, shifted by the clock difference measured on the shared
    events, and contributes the actual value once a print is out.
    Returns (events, shift_h)."""
    shift = calendar_time_check(jb_events, ff_events)
    # Pair again on clock-corrected times: with the offset removed each
    # occurrence lands on its own counterpart rather than on its neighbour.
    actual_by_ff = {}
    used = set()
    for i, j, _ in _pair_events(jb_events, ff_events, shift or 0.0):
        used.add(i)
        actual_by_ff[j] = jb_events[i]["actual"]
    out = []
    for j, f in enumerate(ff_events):
        e = dict(f)
        if actual_by_ff.get(j) and not e["actual"]:
            e["actual"] = actual_by_ff[j]
        out.append(e)
    delta = datetime.timedelta(hours=shift or 0.0)
    for i, j in enumerate(jb_events):
        if i not in used:
            out.append(dict(j, ts=_iso_z(_cal_parse_ts("iso", j["ts"]) - delta)))
    out.sort(key=lambda e: (e["ts"], e["currency"], e["title"]))
    return out, shift


def _ff_week_end(events):
    """ForexFactory's weekly feed always runs Sunday to Saturday, so its
    coverage reaches that Saturday even when the last event is on Friday."""
    if not events:
        return None
    d = _cal_parse_ts("iso", events[-1]["ts"]).date()
    return (d + datetime.timedelta(days=(5 - d.weekday()) % 7)).isoformat()


def _http_json(url, headers=None, timeout=25):
    h = {"User-Agent": "Mozilla/5.0 (divergence-screener)", "Accept": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _jb_error_text(code, body):
    """JBlanked answers 401 both for a bad key and for an account out of
    credits. The distinction matters: one is a setup mistake, the other is a
    working key that simply cannot fetch until credits are topped up."""
    msg = ""
    try:
        msg = (json.loads(body) or {}).get("message", "")
    except Exception:
        msg = (body or "")[:200]
    if "credit" in msg.lower():
        return f"HTTP {code}: {msg.strip()}"
    if code == 401:
        return f"HTTP 401: key rejected ({msg.strip()})" if msg else "HTTP 401: key rejected"
    return f"HTTP {code}: {msg.strip()}" if msg else f"HTTP {code}"


def fetch_jblanked(key):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    url = (f"{JB_CAL_URL}?from={today - datetime.timedelta(days=1)}"
           f"&to={today + datetime.timedelta(days=CAL_DAYS_AHEAD)}&offset={JB_TIME_OFFSET}")
    try:
        data = _http_json(url, {"Authorization": "Api-Key " + key,
                                "Content-Type": "application/json"})
    except urllib.error.HTTPError as e:
        raise RuntimeError(_jb_error_text(e.code, e.read().decode("utf-8", "replace"))) from None
    if isinstance(data, dict):
        if data.get("message") and not data.get("results"):
            raise RuntimeError(str(data["message"]))
        data = data.get("results") or data.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError("unexpected response shape")
    return data


def fetch_ff():
    data = _http_json(FF_CAL_URL)
    if not isinstance(data, list):
        raise RuntimeError("unexpected response shape")
    return data


def refresh_calendar(force=False):
    """Pull the calendar when the cache is stale. A failed fetch never blanks
    the last good list; it lands in meta["error"] and the tab's audit line."""
    now = datetime.datetime.now(datetime.timezone.utc)
    with _lock:
        attempt = _cal_state["attempt"]
        if attempt is not None:
            age = (now - attempt).total_seconds()
            if age < CAL_MIN_REFRESH_S or (not force and age < CAL_REFRESH_MIN * 60):
                return
        _cal_state["attempt"] = now

    key = jb_api_key()
    errors, ff_events = [], []
    # ForexFactory is fetched every time: it is free, authoritative for this
    # week, the clock reference for JBlanked, and the fallback.
    ff_ok = False
    try:
        ff_events = normalise_events(fetch_ff(), "forexfactory")
        ff_ok = True
    except Exception as e:
        errors.append(f"ForexFactory: {e}")

    # JBlanked costs a credit per call, so it keeps its own cache and its own
    # much longer interval; in between, the last pull still extends the horizon.
    jb_fresh = False
    with _lock:
        jb_cache = _cal_state.setdefault("jb", {"events": [], "ts": None, "attempt": None})
        jb_events = list(jb_cache["events"])
        jb_attempt = jb_cache["attempt"]
    if key:
        due = (jb_attempt is None or
               (now - jb_attempt).total_seconds() >= CAL_JB_REFRESH_MIN * 60)
        if due:
            with _lock:
                _cal_state["jb"]["attempt"] = now
            try:
                jb_events = normalise_events(fetch_jblanked(key), "jblanked")
                jb_fresh = True
                with _lock:
                    _cal_state["jb"]["events"] = jb_events
                    _cal_state["jb"]["ts"] = now
            except Exception as e:
                errors.append(f"JBlanked: {e}")   # cached events stay in jb_events

    # Nothing new arrived, so there is nothing better to publish than what is
    # already stored. Rebuilding from the JBlanked cache alone would quietly
    # drop every ForexFactory event from the last good result.
    if not ff_ok and not jb_fresh:
        with _lock:
            meta = dict(_cal_state["meta"])
            meta["error"] = "; ".join(errors) if errors else "no data returned"
            _cal_state["meta"] = meta
        save_calendar()
        return

    with _lock:
        last_shift = (_cal_state["meta"] or {}).get("time_check_h")

    shift, stale_shift = None, False
    jb_end = (datetime.datetime.now(datetime.timezone.utc).date()
              + datetime.timedelta(days=CAL_DAYS_AHEAD)).isoformat()
    if jb_events and ff_events:
        events, shift = merge_calendars(ff_events, jb_events)
        source = "jblanked+forexfactory"
        covered_to = max(jb_end, _ff_week_end(ff_events) or "")
    elif jb_events:
        # No ForexFactory means no clock reference this run. JBlanked's raw
        # times are hours off, so reuse the last measured shift rather than
        # publishing a countdown that is silently wrong.
        shift, stale_shift = last_shift, last_shift is not None
        delta = datetime.timedelta(hours=shift or 0.0)
        events = sorted((dict(e, ts=_iso_z(_cal_parse_ts("iso", e["ts"]) - delta))
                         for e in jb_events),
                        key=lambda e: (e["ts"], e["currency"], e["title"]))
        source, covered_to = "jblanked", jb_end
    elif ff_events:
        events, source = ff_events, "forexfactory"
        covered_to = _ff_week_end(ff_events)
    else:
        events, source, covered_to = [], None, None

    meta = {"source": source, "has_key": bool(key),
            "error": "; ".join(errors) if errors else None,
            "time_check_h": shift, "covered_to": covered_to, "fallback": None}
    if source == "forexfactory":
        meta["fallback"] = ("No JBlanked key set" if not key
                            else "JBlanked failed, using the ForexFactory feed")
    elif source == "jblanked":
        meta["fallback"] = (
            f"ForexFactory is down; JBlanked times were corrected by the last "
            f"measured clock offset ({shift:+g}h)." if stale_shift else
            "ForexFactory is down, so JBlanked's clock could not be checked "
            "against anything. Times may be off by hours.")
    with _lock:
        if events:
            _cal_state["events"] = events
            _cal_state["ts"] = now
            _cal_state["meta"] = meta
        else:
            old = dict(_cal_state["meta"])
            old["error"] = meta["error"] or "no data returned"
            _cal_state["meta"] = old
    save_calendar()


def save_calendar():
    try:
        with _lock:
            jb = _cal_state.get("jb") or {}
            data = {"events": list(_cal_state["events"]),
                    "ts": _cal_state["ts"].isoformat() if _cal_state["ts"] else None,
                    "meta": dict(_cal_state["meta"]),
                    "jb": {"events": list(jb.get("events") or []),
                           "ts": jb["ts"].isoformat() if jb.get("ts") else None}}
        with open(CAL_STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[calendar] save failed: {e}")


def load_calendar():
    try:
        with open(CAL_STATE_FILE) as f:
            data = json.load(f)
        ts = data.get("ts")
        with _lock:
            _cal_state["events"] = data.get("events", [])
            _cal_state["ts"] = datetime.datetime.fromisoformat(ts) if ts else None
            _cal_state["attempt"] = _cal_state["ts"]   # cooldown survives a restart
            _cal_state["meta"] = data.get("meta", {})
            jb = data.get("jb") or {}
            jb_ts = jb.get("ts")
            # The credit already spent on this pull survives a restart too.
            _cal_state["jb"] = {"events": jb.get("events", []),
                                "ts": datetime.datetime.fromisoformat(jb_ts) if jb_ts else None,
                                "attempt": datetime.datetime.fromisoformat(jb_ts) if jb_ts else None}
        print(f"[calendar] restored {len(_cal_state['events'])} event(s)")
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[calendar] load failed: {e}")


# ── Rate limiting + fetching ───────────────────────────────────────────────────
class RateLimiter:
    """Shared token bucket across worker threads.

    ccxt's own throttle is per-instance, and each worker needs its own instance
    (the sync throttle isn't thread-safe). N instances each pacing themselves at
    the exchange limit would send N× the allowed rate, so instance throttling is
    off and every request passes through this one limiter instead.
    """
    def __init__(self, rps):
        self.interval = 1.0 / max(0.5, float(rps))
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_at)
            self.next_at = t + self.interval
        delay = t - time.monotonic()
        if delay > 0:
            time.sleep(delay)


_TRANSIENT = ("429", "Too Many", "too many", "Too many", "10006", "rate limit",
              "50011", "timeout", "timed out", "Connection", "temporarily",
              "Service", "502", "503", "504",
              # Windows AV/proxy setups intermittently MITM TLS and the whole
              # scan dies as "no data" — these are retryable, not fatal
              "certificate", "SSL", "EOF occurred")


def fetch_ohlcv(ex, limiter, sym, tf, limit=300, params=None, since=None, tries=3):
    """OHLCV fetch with backoff. The old bare `except: return None` silently
    dropped rate-limited pairs mid-scan, which quietly shrinks the universe you
    think you scanned."""
    for attempt in range(tries):
        try:
            limiter.acquire()
            return ex.fetch_ohlcv(sym, timeframe=tf, limit=limit,
                                  since=since, params=params or {})
        except Exception as e:
            msg = str(e)
            if any(t in msg for t in _TRANSIENT) and attempt < tries - 1:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
    return None


def _to_df(raw):
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def fetch_df_deep(ex, limiter, sym, tf, target_bars):
    """Fetch up to target_bars by paginating BACKWARD from now.

    Why bother: EWM indicators (MACD/ATR) on a 300-bar window have not
    converged near the window start, so the same chart yields different pivot
    MACD values than a full-history calculation. OKX's plain candles endpoint
    caps at ~300 recent bars; passing `after` (an older-than cursor) walks
    further back. Binance/Bybit honour large limits and usually finish in one
    call.
    """
    first = 300 if ex.id == "okx" else min(target_bars, 1000)
    df = _to_df(fetch_ohlcv(ex, limiter, sym, tf, limit=first))
    if df is None or df.empty:
        return df

    tf_ms = ex.parse_timeframe(tf) * 1000
    for _ in range(8):
        if len(df) >= target_bars:
            break
        oldest = int(df.index[0].timestamp() * 1000)
        if ex.id == "okx":
            raw = fetch_ohlcv(ex, limiter, sym, tf, limit=300, params={"after": oldest})
        else:
            raw = fetch_ohlcv(ex, limiter, sym, tf, limit=1000,
                              since=max(oldest - tf_ms * 1000, 0))
        older = _to_df(raw)
        if older is None or older.empty:
            break
        merged = pd.concat([older, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if len(merged) <= len(df):
            break                      # no progress → start of history
        df = merged
    return df


def drop_forming_candle(df, tf_minutes):
    """Remove the candle that is still open.

    ccxt returns the in-progress bar as the last row. Leaving it in means MACD
    and ATR are computed partly from a bar that will change, so a scan run at
    14:05 and one at 15:55 can disagree about the same setup. A signal needs a
    closed cross, so the open bar cannot create one, but it can still shift the
    indicator values.
    """
    if df is None or df.empty:
        return df
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    if df.index[-1] + pd.Timedelta(minutes=tf_minutes) > pd.Timestamp(now):
        return df.iloc[:-1]
    return df


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


def load_venue(exid):
    """Markets, active USDT perps and bulk tickers for one exchange. Raises when
    the exchange id is not allowed or markets / tickers fail to load."""
    ex = make_exchange(exid)
    mkts = ex.load_markets()
    return {"ex": ex, "mkts": mkts, "perps": usdt_perps(mkts), "tickers": ex.fetch_tickers()}


def fetch_btc(ex, limiter, bars=TF_CFG["4h"]["bars"]):
    """BTC 4H RSI 14, close and 200 EMA on completed bars, for the BTC state.
    None when unavailable."""
    df = drop_forming_candle(
        fetch_df_deep(ex, limiter, "BTC/USDT:USDT", "4h", bars), 240)
    if df is None or len(df) < 80:
        return None
    return pd.DataFrame({"rsi": rsi(df["close"]), "close": df["close"],
                         "ema200": ema(df["close"], EMA_TREND)}).dropna()


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


# ── Dedupe ─────────────────────────────────────────────────────────────────────
def dedupe(sigs):
    """One signal per side per pair.

    Successive divergences of the same side are the same setup seen twice. Keep the
    newest that is still alive; if every one on that side is invalidated, keep
    the newest anyway so the pair still reports what happened rather than going
    silent.
    """
    best = {}
    for s in sigs:
        k = s["side"]
        cur = best.get(k)
        if cur is None:
            best[k] = s
            continue
        live_new = s["status"] != "invalidated"
        live_cur = cur["status"] != "invalidated"
        if live_new != live_cur:
            if live_new:
                best[k] = s
        elif s["bars_ago"] < cur["bars_ago"]:
            best[k] = s
    return sorted(best.values(), key=lambda s: s["bars_ago"])


_VERDICT_RANK = {"enter": 0, "wait": 1, "skip": 2}


def build_rows(sigs, mover=None, flow=None, context_error=None, blocker=None, macro=None):
    """One row per side for ONE pair's signals across timeframes.

    The primary signal — the one whose status, levels and entry verdict the
    table shows — is the one with the best entry verdict (enter, then wait,
    then skip); on a tie the earlier timeframe in TF_CFG order (4H) wins.
    `blocker` / `macro` are the two halves of macro_window(): the first feeds
    the verdict, the second only becomes the row's MACRO tag. A blocker due
    within 24h that does not change the primary's answer (outside its wait
    window, or the answer is already No / Not yet for another reason) is the
    MACRO tag too; the sooner of the two is shown.
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
        verdicts = {tf: entry_verdict(r["signals"][tf], blocker)
                    for tf in r["tfs"]}
        best = min(r["tfs"], key=lambda tf: _VERDICT_RANK[verdicts[tf]["verdict"]])
        p = r["signals"][best]
        r["primary_tf"] = best
        r["score"], r["grade"], r["regime"] = p["score"], p["grade"], p["regime"]
        r["fresh"]  = any(s["fresh"] for s in r["signals"].values())
        r["status"] = p["status"]
        r["entry"]  = verdicts[best]
        due = [macro] if macro else []
        if blocker and 0 < blocker[0] <= 24 and verdicts[best] == entry_verdict(p, None):
            due.append(blocker)
        r["macro"] = f"MACRO {min(due)[1]} in {_hours_txt(min(due)[0])}" if due else None
    return list(rows.values())


def detections(df, forming_df, rules, sides=("bullish", "bearish")):
    """Every detector hit on the completed frame, plus a hit whose signal bar is
    the still-open candle of the forming frame, as (frame, forming, hit)."""
    for src, forming in ((forming_df, True), (df, False)):
        if src is None:
            continue
        for side in sides:
            for d in detect_wave_divergence(src, side, rules):
                if not forming or d["signal_bar"] == len(src) - 1:
                    yield src, forming, d


def venue_confirms(sig, df, forming_df, rules):
    """True when another venue's frames show a signal on the same side whose
    signal bar is within one bar of `sig`'s (the same detection the scan runs)."""
    if df is None or len(df) < 80:
        return False
    bar_ms = TF_CFG[sig["tf"]]["minutes"] * 60_000
    return any(abs(int(src.index[d["signal_bar"]].timestamp() * 1000) - sig["signal_ms"]) <= bar_ms
               for src, _, d in detections(df, forming_df, rules, (sig["side"],)))


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
    for src, forming, d in detections(df, forming_df, rules):
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


# ── Chart preview ──────────────────────────────────────────────────────────────
def _plain(v):
    """A price as plain digits for a chart label (no exponent on sub-cent coins)."""
    return np.format_float_positional(v, trim="-")


def render_chart(df, d, sig):
    """PNG of the A-to-B leg: candles with the entry, stop and R lines (each
    labelled with its price) over the MACD panel with its trough and crossover
    dots; A and B carry their price and MACD value. `d` is the raw detection
    (bar indexes into `df`), `sig` the finished signal (levels)."""
    lo = max(0, d["a_start"] - 10)
    v = df.iloc[lo:min(len(df), d["signal_bar"] + 30)]
    x = np.arange(len(v))
    bg, fg, mut, line = "#0C111B", "#DCE3EE", "#9AA8BD", "#222D40"      # the page's tokens
    green, red, amber, acc = "#6FCF9F", "#E58C98", "#E2B866", "#8FB4E8"
    fig = Figure(figsize=(6.4, 4.6), tight_layout=True, facecolor=bg)   # shown about 460px wide
    ax, am = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for a in (ax, am):
        a.set_facecolor(bg)
        a.tick_params(colors=mut, labelsize=9)
        for sp in a.spines.values():
            sp.set_color(line)

    def ab(a, ia, ib, ya, yb, fmt):
        """The A-to-B line with each end labelled with its value, on the side
        away from the line (below the troughs of a bullish one, above peaks)."""
        a.plot([ia - lo, ib - lo], [ya, yb], color=amber, lw=1.5, marker="o")
        dy, va = (-12, "top") if d["side"] == "bullish" else (8, "bottom")
        for name, i, y in (("A", ia, ya), ("B", ib, yb)):
            a.annotate(f"{name} {fmt(y)}", (i - lo, y), xytext=(0, dy), textcoords="offset points",
                       color=fg, fontsize=10, ha="center", va=va)

    up = (v["close"] >= v["open"]).values
    for mask, col in ((up, green), (~up, red)):
        ax.vlines(x[mask], v["low"].values[mask], v["high"].values[mask], color=col, lw=1)
        ax.vlines(x[mask], v["open"].values[mask], v["close"].values[mask], color=col, lw=3)
    ab(ax, d["a_p"], d["b_p"], d["a_price"], d["b_price"], lambda y: _plain(px(y)))
    ax.margins(y=0.15)      # room for the A / B labels
    am.margins(y=0.2)
    t = sig["targets"]
    for lvl, col, ls, lbl in ((sig["entry_ref_price"], acc, "-", "Entry"),
                              (sig["stop_price"], red, "--", "Stop"), (t["1r"], mut, ":", "1R"),
                              (t["2r"], mut, ":", "2R"), (t["3r"], mut, ":", "3R")):
        ax.axhline(lvl, color=col, lw=0.8, ls=ls)
        ax.text(1.01, lvl, f"{lbl} {_plain(lvl)}", color=col, fontsize=10, va="center",
                transform=ax.get_yaxis_transform())      # right of the panel, clear of the candles
    ax.set_title(f"{sig['base']} {TF_CFG[sig['tf']]['label']} {d['side']} divergence, "
                 f"grade {sig['grade']} ({sig['score']})", fontsize=11, color=fg)

    am.plot(x, v["macd"].values, color=acc, lw=1.2, label="MACD")
    am.plot(x, v["signal"].values, color=mut, lw=1, label="signal")
    am.axhline(0, color=line, lw=0.8)
    ab(am, d["a_m"], d["b_m"], d["a_macd"], d["b_macd"], lambda y: f"{y:.3g}")
    for bar, direction in d["crosses"]:
        am.plot(bar - lo, v["signal"].values[bar - lo], "o", ms=4,
                color=green if direction > 0 else red)
    am.legend(loc="upper left", fontsize=9, facecolor=bg, edgecolor=line, labelcolor=mut)
    step = max(1, len(v) // 4)
    shown = v.index[::step] + pd.Timedelta(hours=DISPLAY_TZ_OFFSET_H)
    am.set_xticks(x[::step])
    am.set_xticklabels([ts.strftime("%b %d %H:%M") for ts in shown])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


# ── Backtest ───────────────────────────────────────────────────────────────────
BT_BARS, BT_HTF_BARS_1D = 1500, 300     # deeper than the live scan, so the replay after MIN_HISTORY is long
BT_SLIP, BT_FEE = 0.0005, 0.001          # slippage against the entry; round-trip fee, both of price
BT_TIME_EXIT = {"4h": 30, "1h": 48}      # bars after entry, then out at the close
BT_IN_SAMPLE = 0.7                       # first 70% of each frame's replayed bars
BT_MIN_N = 30                            # fewer trades than this is "insufficient"


def trade_result(df, i, side, entry, stop, target, max_bars):
    """One exit policy for a trade entered at the close of bar i: +target R at
    the target, -1R at the stop (checked first when one bar touches both), or
    the close max_bars later marked to market. The round-trip fee is taken off
    in R. None when the history ends before any of the three."""
    k = 1 if side == "bullish" else -1
    risk = max((entry - stop) * k, entry * 0.005)
    hi, lo, cl = df["high"].values, df["low"].values, df["close"].values
    mae = 0.0
    for j in range(i + 1, min(i + max_bars + 1, len(df))):
        worst, best = (lo[j], hi[j]) if k > 0 else (hi[j], lo[j])
        mae = max(mae, (entry - worst) * k / risk)
        if (worst - stop) * k <= 0:
            r = -1
        elif (best - entry) * k >= target * risk:
            r = target
        elif j - i == max_bars:
            r = (cl[j] - entry) * k / risk
        else:
            continue
        return {"r": round(float(r - entry * BT_FEE / risk), 3), "bars": j - i,
                "mae_r": round(float(mae), 2)}
    return None


def backtest_pair(df, tf, rules, htf=None, btc=None):
    """Replay one pair bar by bar. Bar i sees df.iloc[:i + 1] and nothing else.

    A signal is taken the first time the detector reports it, at that bar's
    close. Not replayed: open interest (no bar-by-bar history), so that
    component leaves the score's denominator, and the lower-timeframe trigger.

    The first replayed bar is the first one the live scan would analyse
    (MIN_HISTORY completed bars). The last is BT_TIME_EXIT bars before the
    end, so every trade taken resolves under all three exits: a later entry
    would only be kept when it hit the stop or a target early.
    """
    cfg, seen, trades = TF_CFG[tf], set(), []
    close = df["close"].values
    first, last = MIN_HISTORY - 1, len(df) - 1 - BT_TIME_EXIT[tf]
    cut = first + BT_IN_SAMPLE * (last + 1 - first)
    for i in range(first, last + 1):
        view = df.iloc[:i + 1]
        for side in ("bullish", "bearish"):
            k = 1 if side == "bullish" else -1
            for d in detect_wave_divergence(view, side, rules):
                key = (side, d["b_m"])
                if d["signal_bar"] <= i - WHIP_BARS or key in seen:
                    continue                       # old news, or already taken
                seen.add(key)
                sig = _mk(view, d, cfg)
                entry, stop = float(close[i]) * (1 + k * BT_SLIP), sig["stop_price"]
                if (entry - stop) * k <= 0:
                    continue                       # already through the stop when first seen
                # Graded as of the signal bar's close, exactly like add_context.
                close_ts = pd.Timestamp(sig["signal_close_ms"], unit="ms")
                state = btc_state_at(btc, close_ts)
                score, grade, _ = score_signal(sig, htf_agrees(htf, close_ts, cfg["htf_minutes"], side),
                                               btc_state=state)
                res = {n: trade_result(df, i, side, entry, stop, n, BT_TIME_EXIT[tf]) for n in (1, 2, 3)}
                trades.append(dict(
                    {f"r{n}": o and o["r"] for n, o in res.items()},
                    bars=res[2] and res[2]["bars"], mae_r=res[2] and res[2]["mae_r"],
                    tf=tf, side=side, entry=entry, in_sample=i < cut,
                    score=score, grade=grade, time=str(view.index[-1]),
                    regime=regime_tag(side, sig["above_ema200"], state)))
    return trades


def group_stats(trades):
    """Statistics over the trades resolved under the 2R policy (they are then
    resolved under 1R too). Below BT_MIN_N only the count is returned."""
    ts = sorted((t for t in trades if t["r2"] is not None), key=lambda t: t["time"])
    if len(ts) < BT_MIN_N:
        return {"n": len(ts)}
    r1, r2 = [t["r1"] for t in ts], [t["r2"] for t in ts]
    loss = -sum(r for r in r2 if r < 0)
    streak = run = 0
    for r in r2:
        run = run + 1 if r <= 0 else 0
        streak = max(streak, run)
    return {"n": len(ts),
            "win1": round(100 * sum(r > 0 for r in r1) / len(ts), 1),
            "win2": round(100 * sum(r > 0 for r in r2) / len(ts), 1),
            "exp1": round(statistics.mean(r1), 2), "exp2": round(statistics.mean(r2), 2),
            "pf": round(sum(r for r in r2 if r > 0) / loss, 2) if loss else float("inf"),
            "bars": statistics.median(t["bars"] for t in ts),
            "mae": round(statistics.mean(t["mae_r"] for t in ts), 2), "streak": streak}


def tune_cutoff(trades):
    """Grade A cutoff on the 2R policy: the lowest in-sample score whose trades
    at or above it number BT_MIN_N+ with expectancy >= +0.3R, kept only if the
    out-of-sample trades at that cutoff earn > 0 and at least half as much.
    Returns (cutoff or None, the sentence to print)."""
    done = [t for t in trades if t["r2"] is not None]
    ins = [t for t in done if t["in_sample"]]
    for c in sorted({int(t["score"]) for t in ins}):
        a = [t["r2"] for t in ins if t["score"] >= c]
        if len(a) >= BT_MIN_N and statistics.mean(a) >= 0.3:
            break
    else:
        return None, (f"Grade cutoffs: {GRADE_A} / {GRADE_B} kept - no in-sample score has "
                      f"{BT_MIN_N}+ trades at or above it with expectancy >= +0.3R at 2R.")
    ea = statistics.mean(a)
    b = [t["r2"] for t in done if not t["in_sample"] and t["score"] >= c]
    eb = statistics.mean(b) if b else None
    seen = (f"in-sample n={len(a)} {ea:+.2f}R, out-of-sample n={len(b)} "
            + (f"{eb:+.2f}R" if b else "no trades"))
    if b and eb > 0 and eb >= ea / 2:
        return c, (f"Grade cutoffs: recommend grade A at score >= {c} ({seen}). "
                   f"GRADE_A is not changed by the backtest.")
    return None, (f"Grade cutoffs: {GRADE_A} / {GRADE_B} kept - score >= {c} qualified in-sample "
                  f"but did not hold out of sample ({seen}; needs > 0 and >= half the in-sample value).")


def run_backtest(exid="binance", n_pairs=30, symbols=(), strictness="balanced"):
    """Fetch the scan's own history for the most liquid pairs and print the
    statistics per timeframe x grade x regime, in-sample next to out-of-sample,
    then the grade cutoff check and the known gaps."""
    rules = STRICTNESS[strictness]
    ex = make_exchange(exid)
    mkts = ex.load_markets()
    limiter = RateLimiter(SPEED["safe"]["rps"][exid])
    perps = usdt_perps(mkts)
    if symbols:
        pairs = [s for s in perps if s.split("/")[0] in symbols]
        if not pairs:
            print(f"No matching USDT perps on {exid.upper()}.")
            return
    else:
        tickers = ex.fetch_tickers()
        pairs = sorted(perps, key=lambda s: usd_volume(tickers.get(s)) or 0, reverse=True)[:n_pairs]

    def load(sym, tf, bars, minutes):
        df = drop_forming_candle(fetch_df_deep(ex, limiter, sym, tf, bars), minutes)
        return with_indicators(df) if df is not None and len(df) >= 80 else None

    btc = fetch_btc(ex, limiter, BT_BARS)
    trades = []
    for n, sym in enumerate(pairs, 1):
        d4 = load(sym, "4h", BT_BARS, 240)
        d1h = load(sym, "1h", BT_BARS, 60)
        d1d = load(sym, "1d", BT_HTF_BARS_1D, 1440)
        got = (backtest_pair(d4, "4h", rules, d1d, btc) if d4 is not None else []) + \
              (backtest_pair(d1h, "1h", rules, d4, btc) if d1h is not None else [])
        trades += got
        print(f"[{n}/{len(pairs)}] {sym}: {len(got)} signal(s)")

    def half(s):
        if "pf" not in s:
            return f"{s['n']:>4}  insufficient".ljust(55)
        return (f"{s['n']:>4}{s['win1']:>7.1f}{s['win2']:>7.1f}{s['exp1']:>+7.2f}{s['exp2']:>+7.2f}"
                f"{s['pf']:>6.2f}{s['bars']:>6.1f}{s['mae']:>6.2f}{s['streak']:>5}")

    cols = f"{'n':>4}{'win1R':>7}{'win2R':>7}{'exp1R':>7}{'exp2R':>7}{'PF':>6}{'bars':>6}{'MAE':>6}{'lose':>5}"
    print(f"\nREG_DIV, {len(trades)} signals on {len(pairs)} pairs ({exid.upper()}, {strictness})")
    for tf in ("4h", "1h"):
        ts = [t for t in trades if t["tf"] == tf]
        groups = [("ALL", ts)] + [(f"{g} {r}", [t for t in ts if (t["grade"], t["regime"]) == (g, r)])
                                  for g, r in sorted({(t["grade"], t["regime"]) for t in ts})]
        print(f"\n{TF_CFG[tf]['label']:<18}{'in-sample (first 70%)':<55} | out-of-sample (last 30%)")
        print(f"{'grade regime':<18}{cols:<55} | {cols}")
        for label, g in groups:
            print(f"{label:<18}{half(group_stats([t for t in g if t['in_sample']]))} | "
                  f"{half(group_stats([t for t in g if not t['in_sample']]))}")
    print()
    for tf in ("4h", "1h"):        # per timeframe: their in / out-of-sample windows are different dates
        print(f"{TF_CFG[tf]['label']} {tune_cutoff([t for t in trades if t['tf'] == tf])[1]}")
    print(f"""
n = trades resolved under the 2R exit. winNR / expNR = share above zero and average R after
fees when exiting at the stop, at N x risk, or at the close {BT_TIME_EXIT['4h']} bars (4H) /
{BT_TIME_EXIT['1h']} bars (1H) later. PF, bars (median to outcome), MAE (average, in R) and lose
(longest losing streak, by time) use the 2R exit. Entry: the close of the bar the signal first
appears on, {BT_SLIP:.2%} slippage against the trade; fees {BT_FEE:.2%} of price round trip, in R.
A bar touching both the stop and the target counts as the stop. Fewer than {BT_MIN_N} -> insufficient.
Replay: {BT_BARS} bars per timeframe, from the {MIN_HISTORY}th (the live minimum) to {BT_TIME_EXIT['4h']} (4H) /
{BT_TIME_EXIT['1h']} (1H) bars before the end, so every trade resolves under all three exits.

Known gaps: open interest is not replayed (no bar-by-bar history), so that component is dropped
and the score rescaled, as live does when OI is missing. The lower-timeframe triggers are not
replayed, and neither are the scan's pair filters: the weekly volume floor, the 4H ATR% range
and the spread / OI thin-coin filters (pairs are the most liquid now). Funding is not replayed.""")


# ── State persistence ──────────────────────────────────────────────────────────
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
        if meta.get("v") != STATE_VERSION:
            # Written with an older row shape. The page cannot draw it, so
            # start empty rather than broken.
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


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/")
def index():
    # Thresholds are templated in so the dashboard's highlights and tooltips
    # cannot drift from the constants the scans actually use.
    tmpl = {
        "__TF_CFG_JSON__": json.dumps({k: {"label": v["label"], "minutes": v["minutes"],
                                           "fresh_h": v["fresh_h"], "macro_h": MACRO_WAIT_H[k]}
                                       for k, v in TF_CFG.items()}),
        "__DEFAULT_VOL_M__": str(int(DEFAULT_MIN_WEEKLY_VOL / 1e6)),
        "__DEFAULT_MAX_PAIRS__": str(DEFAULT_MAX_PAIRS),
        "__MOVER_CHG__":   format(MOVER_MIN_CHG_PCT, "g"),
        "__MOVER_VOL__":   format(MOVER_VOL_MULT, "g"),
        "__MOVER_RNG__":   format(MOVER_RANGE_MULT, "g"),
        "__MOVER_EDGE__":  format(MOVER_EDGE_PCT, "g"),
        "__MOVER_THIN_M__": str(int(MOVER_THIN_MIN_WEEKLY_VOL / 1e6)),
        "__OI_VOL__":      format(OI_VOL_RATIO_HIGH, "g"),
        "__FUND_SPOT__":   format(FUND_SPOT_LED_MAX, "g"),
        "__FUND_CROWD__":  format(FUND_CROWDED_MIN, "g"),
        "__FUND_SQUEEZE__": format(FUND_SQUEEZE_MAX, "g"),
    }
    html = HTML
    for k, v in tmpl.items():
        html = html.replace(k, v)
    return html


@app.route("/signals")
def signals_route():
    with _lock:
        return jsonify({"rows": list(_state["results"]),
                        "meta": dict(_state["meta"]),
                        "ts": _state["ts"].isoformat() if _state["ts"] else None})


@app.route("/chart")
def chart_route():
    hit = _charts.get("|".join(request.args.get(k, "") for k in ("pair", "side", "tf")))
    if hit is None:
        return "No chart data for this row. Run the scan again.", 404
    return Response(render_chart(*hit), mimetype="image/png")


@app.route("/universe")
def universe_route():
    """How big the tradable universe is at a given weekly-volume floor, without
    running a scan. One bulk ticker call — cheap enough to poke at while
    choosing a floor."""
    exid = request.args.get("exchange", "all")
    floor = float(request.args.get("min_weekly_vol", DEFAULT_MIN_WEEKLY_VOL))
    V = {}
    for v in ([PRIMARY_VENUE, "okx", "bybit"] if exid == "all" else [exid]):
        try:
            V[v] = load_venue(v)
        except Exception as e:
            if not V:      # the primary venue is required
                return jsonify({"error": str(e)}), 500

    vols = {(v, s): usd_volume(V[v]["tickers"].get(s)) for v in V for s in V[v]["perps"]}
    rows = []
    for v, sym in coin_venues({v: d["perps"] for v, d in V.items()}, vols,
                              PRIMARY_VENUE if exid == "all" else exid)[1]:
        if vols[(v, sym)]:
            rows.append({"base": sym.split("/")[0], "sym": sym, "venue": v,
                         "wk": vols[(v, sym)] * 7.0})

    rows.sort(key=lambda r: r["wk"], reverse=True)
    passed = [r for r in rows if r["wk"] >= floor]
    return jsonify({
        "exchange": exid,
        "venues": list(V),
        "total_perps": len(rows),
        "passing": len(passed),
        "floor": floor,
        "top": passed[:60],
        "note": "Weekly figure here is 24h x 7 (a fast estimate). The scan "
                "measures true rolling 7-day volume from the candles.",
    })


@app.route("/scan")
def scan_route():
    exid       = request.args.get("exchange", "all")
    strictness = request.args.get("strictness", "balanced")
    speed      = request.args.get("speed", "normal")
    min_vol    = float(request.args.get("min_weekly_vol", DEFAULT_MIN_WEEKLY_VOL))
    max_pairs  = int(request.args.get("max_pairs", DEFAULT_MAX_PAIRS))
    only_syms  = [s.strip().upper() for s in request.args.get("symbols", "").split(",") if s.strip()]

    if strictness not in STRICTNESS:
        strictness = "balanced"
    if speed not in SPEED:
        speed = "normal"

    rules = STRICTNESS[strictness]
    merged = exid == "all"
    primary = PRIMARY_VENUE if merged else exid

    def generate():
        spd = SPEED[speed]
        # Each venue: markets, bulk tickers, funding and its own rate limiter.
        V = {}
        for v in ([PRIMARY_VENUE, "okx", "bybit"] if merged else [exid]):
            yield sse("log", {"msg": f"Loading markets and 24h volumes on {v.upper()}..."})
            try:
                V[v] = d = load_venue(v)
            except Exception as e:
                if v == primary:
                    yield sse("error", {"msg": f"{v.upper()} markets / volumes failed: {e}"}); return
                yield sse("log", {"msg": f"{v.upper()} unavailable, scanning without it: {e}"})
                continue
            if not any(t.get("bid") for t in d["tickers"].values()) and d["ex"].has.get("fetchBidsAsks"):
                # Binance futures tickers carry no top of book: one more bulk call
                # for the spread filter. A failure only leaves the spread unknown.
                try:
                    for s, b in d["ex"].fetch_bids_asks().items():
                        if s in d["tickers"]:
                            d["tickers"][s].update(bid=b.get("bid"), ask=b.get("ask"))
                except Exception:
                    pass
            d["funding"] = fetch_funding_map(d["ex"])
            d["limiter"] = RateLimiter(spd["rps"].get(v, 8))
            yield sse("log", {"msg": f"{len(d['perps'])} active linear USDT perps on {v.upper()}"})
            if not d["funding"]:
                yield sse("log", {"msg": f"No bulk funding data on {v.upper()} — "
                                         "OI-flow health flags will be blank there"})
        main_ex = V[primary]["ex"]
        venues_txt = "/".join(v.upper() for v in V)

        # Nearest USD blocker (for the entry verdict) and nearest other mover
        # (a row tag). A calendar failure must not stop a scan; the verdict
        # then simply ignores the calendar.
        blocker = macro = window = None
        cal_now = datetime.datetime.now(datetime.timezone.utc)
        try:
            refresh_calendar()
            with _lock:
                events = list(_cal_state["events"])
            window = blocker, macro = macro_window(events, cal_now)
        except Exception as e:
            yield sse("log", {"msg": f"Calendar unavailable for the entry verdict: {e}"})

        # Cheap pre-gate on the ticker's 24h volume. It exists only to avoid
        # fetching 700 bars for obviously dead coins — the REAL gate is the
        # 7-day measurement from the candles further down. PREGATE_SLACK is
        # deliberately loose (a third of the daily-average equivalent) because
        # a coin can trade most of its week in one day: a listing that did
        # $190M five days ago and $2M since has a genuinely large 7-day volume
        # and would be thrown away by a tight 24h proxy.
        # Work items are (venue, symbol): one per coin, on the venue coin_venues picks.
        vol24 = {(v, s): usd_volume(V[v]["tickers"].get(s)) for v in V for s in V[v]["perps"]}
        listed, perps = coin_venues({v: d["perps"] for v, d in V.items()}, vol24, primary)
        if merged:
            off = sum(1 for v, _ in perps if v != PRIMARY_VENUE)
            yield sse("log", {"msg": f"{len(perps)} coins across {venues_txt} · {off} not on "
                                     f"{PRIMARY_VENUE.upper()}, scanned on OKX / Bybit"})
        no_ticker = sum(1 for p in perps if not vol24[p])
        if no_ticker:
            yield sse("log", {"msg": f"{no_ticker} perp(s) report no usable 24h volume — not scanned"})

        pre_floor = (min_vol / 7.0) * PREGATE_SLACK
        pairs = [p for p in perps if vol24[p] and vol24[p] >= pre_floor]
        pairs.sort(key=lambda p: vol24[p], reverse=True)
        spread_reject = 0

        if only_syms:
            wanted = {s.split("/")[0].split(":")[0] for s in only_syms}
            pairs = [p for p in perps if p[1].split("/")[0] in wanted]
            found = {p[1].split("/")[0] for p in pairs}
            missing = sorted(wanted - found)
            msg = f"Symbol override: scanning {len(pairs)} pair(s)"
            if missing:
                # Silently ignoring a typo'd or delisted ticker is the worst
                # outcome here — the user reads an empty table as "no setup".
                msg += f" · NOT LISTED on {venues_txt}: {', '.join(missing)}"
            yield sse("log", {"msg": msg})
            if not pairs:
                yield sse("error", {"msg": f"None of those symbols are listed as USDT perps "
                                           f"on {venues_txt}: {', '.join(sorted(wanted))}"})
                return
        else:
            dropped = sum(1 for p in perps if vol24[p]) - len(pairs)
            kept = [p for p in pairs if not spread_too_wide(V[p[0]]["tickers"].get(p[1]))]
            spread_reject, pairs = len(pairs) - len(kept), kept
            if spread_reject:
                yield sse("log", {"msg": f"{spread_reject} pair(s) skipped: spread wider than "
                                         f"{MAX_SPREAD_PCT:g}%"})
            if len(pairs) > max_pairs:
                yield sse("log", {"msg": f"{len(pairs)} pairs cleared the pre-filter — "
                                         f"capping at the {max_pairs} most liquid by 24h volume"})
                pairs = pairs[:max_pairs]
            yield sse("log", {"msg": f"{len(pairs)} candidates · {dropped} too illiquid to be worth "
                                     f"fetching · 7-day volume checked per pair next"})

        btc = fetch_btc(main_ex, V[primary]["limiter"])
        if btc is None:
            yield sse("log", {"msg": "BTC 4H candles unavailable: every setup is tagged COUNTER TREND "
                                     "and scored without the BTC component (PARTIAL DATA)"})
        _charts.clear()
        workers = spd["workers"]
        by_venue = " · ".join(f"{v.upper()} {sum(1 for p in pairs if p[0] == v)}" for v in V)
        yield sse("log", {"msg": f"Scanning {len(pairs)} pairs on 4H + 1H · "
                                 f"{strictness} · {workers} workers" +
                                 (f" ({by_venue})" if merged else "")})
        yield sse("total", {"total": len(pairs)})

        tls = threading.local()

        def worker_ex(v):
            # one ccxt instance per thread per venue (the sync throttle is not thread-safe)
            exs = tls.__dict__.setdefault("exs", {})
            if v not in exs:
                exs[v] = make_exchange(v, rate_limit=False)
                exs[v].set_markets(V[v]["mkts"])     # reuse markets — no extra API call
            return exs[v]

        q = queue.Queue()
        counter = {"n": 0, "analysed": 0, "vol_reject": 0, "no_data": 0, "errors": 0,
                   "no_1h": 0, "atr_reject": 0, "short_history": 0, "oi_reject": 0}
        clock = threading.Lock()
        hist_min = 80 if only_syms else MIN_HISTORY

        def confirm(v, base, rows):
            """Venues (other than v) where the same side fires within one bar of
            each row's primary signal. A venue that fails to fetch confirms nothing."""
            ok = {id(r): [] for r in rows}
            for o, osym in listed[base].items():
                if o == v:
                    continue
                for tf in {r["primary_tf"] for r in rows}:
                    try:
                        raw = fetch_df_deep(worker_ex(o), V[o]["limiter"], osym, tf, TF_CFG[tf]["bars"])
                        if raw is None or len(raw) < 80:
                            continue
                        df = drop_forming_candle(raw, TF_CFG[tf]["minutes"])
                        fdf, df = forming_df(raw, df), with_indicators(df)
                        for r in rows:
                            if r["primary_tf"] == tf and venue_confirms(r["signals"][tf], df, fdf, rules):
                                ok[id(r)].append(o)
                    except Exception:
                        continue
            return ok

        def scan_pair(item):
            v, sym = item
            base = sym.split("/")[0]
            limiter, funding = V[v]["limiter"], V[v]["funding"]
            try:
                ex = worker_ex(v)
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
                wk_vol = weekly_usd_volume(df4, vol24.get(item), TF_CFG["4h"]["minutes"])
                if not only_syms and (wk_vol is None or wk_vol < min_vol):
                    with clock:
                        counter["vol_reject"] += 1
                    return

                f4 = forming_df(raw, df4)
                df4 = with_indicators(df4)
                if len(df4) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return
                if not only_syms and atr_pct_out_of_range(float(df4["atr"].iloc[-1]),
                                                          float(df4["close"].iloc[-1])):
                    with clock:
                        counter["atr_reject"] += 1
                    return

                df1 = f1 = None
                raw1 = fetch_df_deep(ex, limiter, sym, "1h", TF_CFG["1h"]["bars"])
                if raw1 is not None and len(raw1) >= 80:
                    live_price = float(raw1["close"].iloc[-1])    # the newer tick
                    df1 = drop_forming_candle(raw1, TF_CFG["1h"]["minutes"])
                    if df1 is not None:
                        f1 = forming_df(raw1, df1)
                        df1 = with_indicators(df1)
                if df1 is None or len(df1) < 80:
                    with clock:
                        counter["no_1h"] += 1

                # History: a timeframe under hist_min completed bars is not analysed
                # (its frame still serves as the other timeframe's HTF / trigger).
                usable = [d for d in (df4, df1) if d is not None and len(d) >= 80]
                deep = [d for d in usable if len(d) >= hist_min]
                if len(deep) < len(usable):
                    with clock:
                        counter["short_history"] += len(usable) - len(deep)
                if not deep:
                    return
                with clock:
                    counter["analysed"] += 1

                sigs = (signals_for_tf(df4 if len(df4) >= hist_min else None, "4h", rules,
                                       live_price, sym, wk_vol, f4) +
                        signals_for_tf(df1 if df1 is not None and len(df1) >= hist_min else None,
                                       "1h", rules, live_price, sym, wk_vol, f1))
                if not sigs:
                    return
                # Everything below is fetched only for pairs worth a row.
                mover, flow, cerr, d1, oi_d = pair_context(ex, limiter, sym, live_price,
                                                           funding.get(sym), wk_vol)
                if not only_syms and oi_too_thin(oi_d, live_price):
                    with clock:
                        counter["oi_reject"] += 1
                    return
                d1 = with_indicators(d1) if d1 is not None else None
                oi_h = df15 = None
                if any(s["fresh"] for s in sigs):       # stale signals answer No whatever the grade
                    oi_h = fetch_oi_history(ex, limiter, sym, 100, tf="1h")
                    df15 = drop_forming_candle(
                        fetch_df_deep(ex, limiter, sym, "15m", LTF_BARS_15M), 15)
                    df15 = with_indicators(df15) if df15 is not None and len(df15) >= 80 else None
                grade_signals(sigs, df4, df1, d1, df15, oi_d, oi_h, funding.get(sym), btc)
                for s in sigs:
                    _charts[f"{sym}|{s['side']}|{s['tf']}"] = (s.pop("_df"), s.pop("_d"), dict(s))
                rows = build_rows(sigs, mover, flow, cerr, blocker, macro)
                ok = confirm(v, base, rows) if merged else {}
                for row in rows:
                    row["exchange"] = v
                    row["venue_tags"] = venue_tags(v, ok[id(row)]) if merged else []
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

        done = threading.Event()

        def runner():
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(scan_pair, pairs))
            finally:
                done.set()

        threading.Thread(target=runner, daemon=True).start()

        results = []

        def persist():
            # `scanned` used to be len(pairs) — the number DISPATCHED. Pairs
            # that failed the weekly-volume gate, came back with no candles or
            # threw were all counted as scanned, so the headline could be
            # nearly double the number of coins actually analysed. It now
            # reports what reached the detector, and every other bucket
            # is reported alongside it instead of disappearing.
            with clock:
                c = dict(counter)
            meta = {"v": STATE_VERSION, "exchange": exid, "tfs": list(TF_CFG), "strictness": strictness,
                    "min_weekly_vol": min_vol, "universe": len(perps),
                    "dispatched": len(pairs), "scanned": c["analysed"],
                    "vol_rejected": c["vol_reject"], "no_data": c["no_data"],
                    "errors": c["errors"], "no_1h": c["no_1h"], "no_ticker": no_ticker,
                    "venues": list(V), "spread_rejected": spread_reject,
                    "atr_rejected": c["atr_reject"], "short_history": c["short_history"],
                    "oi_rejected": c["oi_reject"],
                    "complete": done.is_set() and q.empty(),
                    **brief_meta(btc, window, cal_now)}
            with _lock:
                _state["results"] = results
                _state["ts"] = datetime.datetime.now(datetime.timezone.utc)
                _state["meta"] = meta
            save_state()
            return meta

        # The finally clause matters: if the browser tab closes mid-scan the
        # generator is torn down at the yield, and v3 lost every result found
        # up to that point. Whatever has been collected gets saved either way.
        try:
            while not (done.is_set() and q.empty()):
                try:
                    kind, payload = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "result":
                    results.append(payload)
                yield sse(kind, payload)
        finally:
            meta = persist()

        yield sse("done", {"hits": len(results), "meta": meta})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/mover_signals")
def mover_signals_route():
    with _lock:
        s = _movers_state
        return jsonify({"rows": list(s["rows"]), "meta": dict(s["meta"]),
                        "ts": s["ts"].isoformat() if s["ts"] else None})


@app.route("/calendar")
def calendar_route():
    refresh_calendar(force=request.args.get("refresh") == "1")
    with _lock:
        return jsonify({"events": list(_cal_state["events"]),
                        "meta": dict(_cal_state["meta"]),
                        "ts": _cal_state["ts"].isoformat() if _cal_state["ts"] else None,
                        "now": _iso_z(datetime.datetime.now(datetime.timezone.utc))})


@app.route("/movers")
def movers_route():
    """Daily movers scan — one SSE stream, ~1 API call per pair.

    Crypto only: is_tradfi() keeps stock / ETF / commodity perps out of the
    universe entirely. A watchlist built at the weekend that quietly includes an
    equity perp is quoting levels on an instrument whose underlying market is
    shut — thin book, funding-driven drift, and a gap at the Monday cash open.
    """
    exid      = request.args.get("exchange", "okx")
    speed     = request.args.get("speed", "normal")
    min_vol   = float(request.args.get("min_weekly_vol", DEFAULT_MIN_WEEKLY_VOL))
    max_pairs = int(request.args.get("max_pairs", DEFAULT_MAX_PAIRS))
    thin      = request.args.get("thin") == "1"
    # Never above the main floor: lowering the floor below $20M already lets
    # those pairs in as ordinary rows, and the band is then simply empty.
    thin_floor = min(MOVER_THIN_MIN_WEEKLY_VOL, min_vol)
    if speed not in SPEED:
        speed = "normal"

    def generate():
        try:
            main_ex = make_exchange(exid)
        except ValueError as e:
            yield sse("error", {"msg": str(e)}); return

        yield sse("log", {"msg": "Loading markets..."})
        try:
            mkts = main_ex.load_markets()
        except Exception as e:
            yield sse("error", {"msg": f"load_markets failed: {e}"}); return

        all_perps = [s for s in mkts
                     if s.endswith("/USDT:USDT") and mkts[s].get("active")
                     and s.split("/")[0] not in STABLE_BASES]
        perps = [s for s in all_perps if not is_tradfi(mkts[s])]
        skipped_tradfi = len(all_perps) - len(perps)
        yield sse("log", {"msg": f"{len(perps)} crypto perps on {exid.upper()} · "
                                 f"{skipped_tradfi} stock/ETF/commodity perp(s) excluded"})

        yield sse("log", {"msg": "Fetching 24h volumes (one bulk call)..."})
        try:
            tickers = main_ex.fetch_tickers()
        except Exception as e:
            yield sse("error", {"msg": f"volume fetch failed: {e}"}); return

        vol24 = {}
        for s in perps:
            v = usd_volume(tickers.get(s))
            if v:
                vol24[s] = v
        pre_floor = ((thin_floor if thin else min_vol) / 7.0) * PREGATE_SLACK
        # Sorted most-liquid first, so the max_pairs cap can only ever cut
        # into the thin end of the list, never the main one.
        pairs = sorted([s for s in vol24 if vol24[s] >= pre_floor],
                       key=lambda s: vol24[s], reverse=True)[:max_pairs]
        yield sse("log", {"msg": f"{len(pairs)} candidates - measuring "
                                 f"{MOVER_LOOKBACK_D}-day volume and range baselines"})

        spd = SPEED[speed]
        limiter = RateLimiter(spd["rps"].get(exid, 8))
        yield sse("total", {"total": len(pairs)})

        tls = threading.local()

        def worker_ex():
            ex = getattr(tls, "ex", None)
            if ex is None:
                ex = make_exchange(exid, rate_limit=False)
                ex.set_markets(mkts)
                tls.ex = ex
            return ex

        q = queue.Queue()
        counter = {"n": 0, "analysed": 0, "vol_reject": 0, "no_data": 0,
                   "quiet": 0, "errors": 0, "thin_analysed": 0, "thin_quiet": 0}
        clock = threading.Lock()

        def scan_pair(sym):
            base = sym.split("/")[0]
            try:
                ex = worker_ex()
                raw = _to_df(fetch_ohlcv(ex, limiter, sym, "1d",
                                         limit=MOVER_LOOKBACK_D + 12))
                if raw is None or raw.empty:
                    with clock:
                        counter["no_data"] += 1
                    return
                live_price = float(raw["close"].iloc[-1])

                # Same ordering trap as the divergence scan: drop the open bar
                # BEFORE any volume measurement. The still-forming day compared
                # against full days makes every pair look quiet.
                df = drop_forming_candle(raw, 1440)
                if df is None or len(df) < MOVER_LOOKBACK_D + 2:
                    with clock:
                        counter["no_data"] += 1
                    return

                wk_vol = weekly_usd_volume(df, vol24.get(sym), 1440)
                tier = "core"
                if wk_vol is None or wk_vol < min_vol:
                    if not (thin and wk_vol is not None and wk_vol >= thin_floor):
                        with clock:
                            counter["vol_reject"] += 1
                        return
                    tier = "thin"

                m = analyse_mover(df, live_price)
                if m is None:
                    with clock:
                        counter["no_data"] += 1
                    return

                with clock:
                    counter["thin_analysed" if tier == "thin" else "analysed"] += 1

                qualifies, flags, score = classify_mover(m, thin=(tier == "thin"))
                if not qualifies:
                    with clock:
                        counter["thin_quiet" if tier == "thin" else "quiet"] += 1
                    return

                m.update({
                    "pair": sym, "base": base,
                    "direction": "UP" if m["chg_1d_pct"] >= 0 else "DOWN",
                    "flags": flags, "score": score,
                    "weekly_volume": round(wk_vol, 2),
                    "daily_volume": round(wk_vol / 7.0, 2),
                    "tier": tier,
                    "current_price": px(live_price),
                })
                q.put(("result", m))
            except Exception as e:
                with clock:
                    counter["errors"] += 1
                q.put(("log", {"msg": f"[{base}] skipped: {e}"}))
            finally:
                with clock:
                    counter["n"] += 1
                    n_now = counter["n"]
                q.put(("progress", {"current": n_now, "pair": sym}))

        done = threading.Event()

        def runner():
            try:
                with ThreadPoolExecutor(max_workers=spd["workers"]) as pool:
                    list(pool.map(scan_pair, pairs))
            finally:
                done.set()

        threading.Thread(target=runner, daemon=True).start()

        rows = []

        def persist():
            with clock:
                c = dict(counter)
            meta = {"exchange": exid, "lookback_d": MOVER_LOOKBACK_D,
                    "range_d": MOVER_RANGE_D, "min_weekly_vol": min_vol,
                    "dispatched": len(pairs), "analysed": c["analysed"],
                    "vol_rejected": c["vol_reject"], "no_data": c["no_data"],
                    "quiet": c["quiet"], "errors": c["errors"],
                    "tradfi_excluded": skipped_tradfi,
                    "thin": thin, "thin_floor": thin_floor if thin else None,
                    "thin_analysed": c["thin_analysed"],
                    "thin_quiet": c["thin_quiet"],
                    "complete": done.is_set() and q.empty()}
            with _lock:
                _movers_state["rows"] = rows
                _movers_state["ts"] = datetime.datetime.now(datetime.timezone.utc)
                _movers_state["meta"] = meta
            save_movers()
            return meta

        # Same discipline as the other two scans: if the browser tab closes
        # mid-scan, whatever was collected is still saved — and flagged
        # incomplete, because a partial result read as "quiet market" is the
        # failure mode that matters.
        try:
            while not (done.is_set() and q.empty()):
                try:
                    kind, payload = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if kind == "result":
                    rows.append(payload)
                yield sse(kind, payload)
        finally:
            meta = persist()

        yield sse("done", {"hits": len(rows), "meta": meta})

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── UI ─────────────────────────────────────────────────────────────────────────
# NOTE: this is a RAW string (r"""). v3 used a normal triple-quoted string, and
# any backslash escape in the JavaScript — \x27, \d in a regex, ’ — got
# eaten by Python before the browser ever saw it. That exact bug shipped once
# and killed the whole script. Raw string, no more class of bug.
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MACD Divergence Screener</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
/* One calm column: brief first, short table second, everything else on click. Inter for words, mono only for numbers. */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  color-scheme:dark;
  --bg:#0C111B;--sf:#121926;--sf2:#182132;--line:#222D40;
  --tx:#DCE3EE;--tx2:#9AA8BD;--tx3:#6E7D94;
  --acc:#8FB4E8;--acc-ink:#0C111B;
  --yes:#6FCF9F;--yes-bg:#12291F;
  --wait:#E2B866;--wait-bg:#2A2212;
  --no:#8E9BB0;--no-bg:#1A2231;
  --short:#E58C98;
  --sans:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',ui-monospace,monospace;
}
body{background:var(--bg);color:var(--tx);font-family:var(--sans);font-size:15px;line-height:1.55;padding:0 16px 64px}
.shell{max-width:1020px;margin:0 auto}
button,select,input{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px;border-radius:4px}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.lbl,.fl,.dash-sec-ttl,.lp-lbl,.cal-next-lbl,.cal-dow{font-size:12px;font-weight:500;letter-spacing:.08em;text-transform:uppercase;color:var(--tx3)}
.calm,.note,.audit,.hint,.wprog,.sig-sub,.learn-sub{font-size:13px;color:var(--tx3)}
.calm,.note,.prose{max-width:68ch}
.prose{color:var(--tx2)}
.prose b,.rule-b b{color:var(--tx);font-weight:500}

.topbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px 20px;padding-block:18px;border-bottom:1px solid var(--line);margin-bottom:28px}
.t-logo{font-weight:600;letter-spacing:.02em}
.nav{display:flex;flex-wrap:wrap;gap:4px}
.nav-btn{background:none;border:0;padding:6px 12px;border-radius:6px;color:var(--tx2)}
.nav-btn:hover{color:var(--tx)}
.nav-btn.active{background:var(--sf2);color:var(--tx)}
.t-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.sdot{width:7px;height:7px;border-radius:50%;background:var(--tx3)}
.sdot.live{background:var(--acc);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
.slbl{font-size:13px;color:var(--tx3);transition:opacity .2s}
.slbl.fd{opacity:.3}
.page{display:none}
.page.active{display:flex;flex-direction:column;gap:28px}

/* ── Brief ── */
.brief{display:flex;flex-direction:column;gap:18px}
.brief h1{font-size:26px;font-weight:600;line-height:1.25;text-wrap:balance}
.brief p{max-width:68ch;color:var(--tx2)}
.brief p b{color:var(--tx);font-weight:500}
.state{display:flex;flex-direction:column;gap:8px;align-items:flex-start}
.picks{display:flex;flex-direction:column;gap:10px}
.pick{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:4px 16px;align-items:center;width:100%;text-align:left;
  background:var(--sf);border:1px solid var(--line);border-radius:10px;padding:14px 16px;transition:border-color .15s}
.pick:hover{border-color:var(--acc)}
.pick .head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px}
.pick .coin{font-size:17px;font-weight:600}
.pick .gr{color:var(--tx2)}
.pick .old{color:var(--tx3);font-size:13px}
.pick .why{grid-column:1/-1;color:var(--tx2);font-size:14px}
.pick .lv{grid-column:1/-1;font-size:13px;color:var(--tx2)}
.pick .lv span{color:var(--tx)}
.side-long{color:var(--yes)}.side-short{color:var(--short)}
.ans{display:inline-block;padding:3px 10px;border-radius:20px;font-size:13px;font-weight:500;white-space:nowrap}
.ans.enter{background:var(--yes-bg);color:var(--yes)}.ans.wait{background:var(--wait-bg);color:var(--wait)}.ans.skip{background:var(--no-bg);color:var(--no)}
table.stale .ans{opacity:.45}

/* ── Scan bar, settings, progress ── */
.scan-sec{display:flex;flex-direction:column;gap:14px}
.bar,.wbar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px}
.rbtn{background:var(--acc);color:var(--acc-ink);border:0;border-radius:8px;padding:9px 22px;font-weight:600;transition:opacity .15s}
.rbtn:hover:not(:disabled){opacity:.88}
.rbtn:disabled,.gbtn:disabled{opacity:.4;cursor:not-allowed}
.gbtn{background:none;color:var(--tx2);border:1px solid var(--line);border-radius:8px;padding:7px 14px;font-size:14px}
.gbtn:hover:not(:disabled){border-color:var(--acc);color:var(--tx)}
.audit.warn{color:var(--wait);background:var(--wait-bg);border-radius:8px;padding:9px 12px}
details.set{margin-left:auto}
details.set[open]{flex-basis:100%;margin-left:0}
details.set summary{cursor:pointer;color:var(--tx2);font-size:14px;padding:6px 4px}
.setbody{margin-top:8px;display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:20px 28px;background:var(--sf);border:1px solid var(--line);border-radius:10px;padding:18px}
.fl{margin-bottom:7px}
.hint{margin-top:7px}
select,input[type=text]{width:100%;background:var(--sf2);border:1px solid var(--line);font-size:14px;padding:8px 10px;border-radius:6px;appearance:none;transition:border-color .2s}
select:focus,input[type=text]:focus{border-color:var(--acc)}
input[type=number]{-moz-appearance:textfield}
input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none}
.nw{display:flex;align-items:center;background:var(--sf2);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.nw:focus-within{border-color:var(--acc)}
.nw input[type=number]{flex:1;min-width:0;background:transparent;border:none;border-left:1px solid var(--line);border-right:1px solid var(--line);text-align:center;padding:8px 4px;font-family:var(--mono);font-size:14px}
.nb{width:38px;height:38px;background:transparent;border:none;color:var(--tx2);font-size:17px}
.nb:hover{background:var(--line);color:var(--tx)}
.trow{display:flex;gap:4px}
.tbtn{flex:1;padding:7px 0;background:var(--sf2);border:1px solid var(--line);color:var(--tx2);font-size:14px;border-radius:6px;transition:color .15s,background .15s}
.tbtn:hover{color:var(--tx)}
.tbtn.active{background:var(--acc);border-color:var(--acc);color:var(--acc-ink);font-weight:600}
.prog{display:flex;flex-direction:column;gap:8px}
.pb-bg{height:3px;background:var(--sf2);border-radius:2px;overflow:hidden}
.pb-fill{height:100%;background:var(--acc);border-radius:2px;width:0%;transition:width 1.4s cubic-bezier(0,.55,.45,1)}
.pb-lbl{font-size:13px;color:var(--tx3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lc{display:flex;flex-direction:column;gap:5px}
.lcard{display:flex;justify-content:space-between;gap:12px;border-left:2px solid var(--line);padding:3px 10px;font-size:13px;transition:opacity .35s,max-height .4s;max-height:60px;overflow:hidden}
.lcard.ce,.lcard.cx{opacity:0}
.lcard.cx{max-height:0;padding-block:0}
.lcard.in{border-left-color:var(--acc)}.lcard.er{border-left-color:var(--short)}.lcard.ol{opacity:.5}
.lcard-m{color:var(--tx2);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lcard.in .lcard-m,.lcard.dn .lcard-m{color:var(--tx)}
.lcard-t{font-family:var(--mono);font-size:12px;color:var(--tx3);flex-shrink:0}
.sl{display:none;position:relative;height:240px;flex-direction:column;align-items:center;justify-content:center;gap:12px;border:1px solid var(--line);border-radius:10px;overflow:hidden;transition:opacity .35s ease}
.sl.vis{display:flex}.sl.diss{opacity:0}
.sbg{position:absolute;inset:0;overflow:hidden;pointer-events:none}
.sbg-feed{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:flex-end;padding:16px 20px;overflow:hidden}
.bgl{font-family:var(--mono);font-size:12px;line-height:1.7;color:var(--tx3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0;opacity:0;transition:opacity .4s ease}
.bgl.sh{opacity:.25}.bgl.sh.hit{opacity:.8;color:var(--tx2)}
.lring{position:relative;width:72px;height:72px}
.lring svg{width:72px;height:72px;transform:rotate(-90deg)}
.rtrack{fill:none;stroke:var(--line);stroke-width:2.5}
.rfill{fill:none;stroke:var(--acc);stroke-width:2.5;stroke-linecap:round;stroke-dasharray:201;stroke-dashoffset:201;transition:stroke-dashoffset 1.4s cubic-bezier(0,.55,.45,1)}
.rpct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:14px;font-weight:600}
.lpair{position:relative;font-size:14px;color:var(--tx2);max-width:220px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.llbl{position:relative;font-size:12px;color:var(--tx3);letter-spacing:.08em;text-transform:uppercase}
.rbar{display:flex;align-items:center;gap:8px 14px;flex-wrap:wrap}
.spill{font-size:13px;color:var(--tx2);display:none}
.spill.vis{display:inline}
.fbar,.sig-filters{display:flex;gap:6px;flex-wrap:wrap}
.fbar{margin-left:auto}
.ftab,.sig-ftab{padding:5px 12px;background:none;border:1px solid var(--line);color:var(--tx2);font-size:13px;border-radius:6px;transition:color .15s,border-color .15s,background .15s}
.ftab:hover,.sig-ftab:hover{color:var(--tx)}
.ftab.active,.sig-ftab.active{background:var(--sf2);color:var(--tx);border-color:var(--tx3)}

/* ── Tables ── */
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{width:100%;border-collapse:collapse;min-width:700px}
th{text-align:left;font-size:12px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;color:var(--tx3);padding:12px 16px;border-bottom:1px solid var(--line);white-space:nowrap}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--tx2)}
th.sorted{color:var(--acc)}
td{padding:14px 16px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
tr.row{cursor:pointer}
tr.row:hover td{background:var(--sf)}
td .sub{display:block;font-size:13px;color:var(--tx3)}
td.none{color:var(--tx3);cursor:default}
td.coin{white-space:nowrap}
td.why{min-width:250px}
.tf{font-size:12px;color:var(--tx3);margin-left:6px}
.tf.on{color:var(--tx2)}
.tag,.bdg,.thin-badge,.cal-imp{display:inline-block;font-size:12px;color:var(--tx2);border:1px solid var(--line);border-radius:5px;padding:1px 7px;margin:4px 4px 0 0;white-space:nowrap}
tr.det td{background:var(--sf);padding:18px 16px}
.dgrid{display:grid;grid-template-columns:minmax(0,1.9fr) minmax(0,1fr) minmax(0,1fr);gap:20px}
.dgrid + .dgrid{margin-top:20px;padding-top:20px;border-top:1px solid var(--line)}
@media (max-width:820px){.dgrid{grid-template-columns:minmax(0,1fr)}}
.dgrid > div{min-width:0;display:flex;flex-direction:column;gap:8px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px}
.kv dt{color:var(--tx3)}
.chk{list-style:none;font-size:14px;display:flex;flex-direction:column;gap:3px}
.chk li::before{content:"\2713";color:var(--yes);margin-right:8px}
.chk li.w::before{content:"\2026";color:var(--wait)}
.chk li.x::before{content:"\00D7";color:var(--short)}
.full{font-size:13px;color:var(--tx2)}
img.chart{display:block;width:100%;border:1px solid var(--line);border-radius:8px}
.fold{background:none;border:0;border-top:1px solid var(--line);color:var(--tx2);padding:12px 16px;width:100%;text-align:left;font-size:14px}
.fold:hover{color:var(--tx)}
.no-sig{color:var(--tx3);padding:20px;text-align:center}

/* ── Movers ── */
.sig-ttl{font-size:20px;font-weight:600}
.wbar select{width:150px}
.mv-thin{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--tx2);cursor:pointer;user-select:none}
.mv-thin input{accent-color:var(--acc)}
.mv-sep td{background:var(--wait-bg);color:var(--wait);font-size:13px;padding:8px 16px}
.thin-badge{color:var(--wait);border-color:var(--wait-bg)}
.mrank{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:5px;background:var(--sf2);font-family:var(--mono);font-size:12px;color:var(--tx2)}
.mrank.top{background:var(--acc);color:var(--acc-ink);font-weight:600}
.mpos{position:relative;width:88px;height:5px;background:var(--sf2);border-radius:3px;margin-top:8px}
.mpos i{position:absolute;top:-3px;width:3px;height:11px;border-radius:2px;background:var(--tx);transform:translateX(-1px)}
.mlv{font-family:var(--mono);font-size:13px;color:var(--tx2);white-space:nowrap}
/* nine columns in the same 1020px column: tighter cells, wrapping headers */
#mvT th,#mvT td{padding:12px 9px}
#mvT th{white-space:normal}
#mvT .sub,td.num .sub{white-space:nowrap}
.mlv small{font-family:var(--sans);color:var(--tx3);font-size:12px}
.qc{display:flex;flex-direction:column;gap:4px;min-width:56px}
.qbar{height:3px;background:var(--sf2);border-radius:2px;overflow:hidden}
.qfill{height:100%;border-radius:2px;background:var(--tx3)}
.hl{color:var(--tx);font-weight:600}
.lo{color:var(--tx2)}

/* ── Calendar ── */
.cal-next,.cal-strip{display:flex;align-items:center;gap:8px 18px;flex-wrap:wrap;background:var(--sf);border:1px solid var(--line);border-radius:10px;padding:14px 18px}
.cal-next.soon,.cal-strip.soon,.cal-day.soon{background:var(--wait-bg)}
.cal-next-lbl{width:100%}
.cal-next-ttl{font-size:17px;font-weight:600}
.cal-next-in,.cal-strip .in{font-family:var(--mono);font-weight:600;color:var(--tx)}
.cal-next.soon .cal-next-in,.cal-strip.soon .in{color:var(--wait)}
.cal-next-meta{font-size:13px;color:var(--tx2)}
.cal-next-meta b,.cal-strip b{color:var(--tx);font-weight:600}
.cal-strip{font-size:14px;color:var(--tx2);padding:10px 16px}
.cal-strip a,.cal-out{color:var(--tx2);font-size:13px;text-decoration:none;cursor:pointer;white-space:nowrap}
.cal-strip a{margin-left:auto}
.cal-strip a:hover,.cal-out:hover{color:var(--acc)}
.cal-grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:8px}
.cal-day{background:var(--sf);border:1px solid var(--line);border-radius:8px;padding:10px;min-height:110px;cursor:pointer;display:flex;flex-direction:column;gap:6px;transition:border-color .15s}
.cal-day:hover{border-color:var(--tx3)}
.cal-day.past{opacity:.4}
.cal-day.today{border-color:var(--acc)}
.cal-day.sel{background:var(--sf2);border-color:var(--tx3)}
.cal-dh{display:flex;justify-content:space-between;align-items:baseline}
.cal-dn{font-family:var(--mono);font-size:14px;font-weight:600}
.cal-day.today .cal-dn{color:var(--acc)}
.cal-ev{font-size:12px;line-height:1.35;background:var(--sf2);border-left:2px solid var(--line);border-radius:4px;padding:5px 7px;overflow-wrap:anywhere}
.cal-ev.mv{border-left-color:var(--acc)}
.cal-ev.done{opacity:.5}
.cal-ev-t{font-family:var(--mono);color:var(--tx2);margin-right:5px}
.cal-ev-c{color:var(--tx3);margin-left:4px}
.cal-quiet{font-size:12px;color:var(--tx3);margin-top:auto}
.cal-imp{margin:0}
.cal-imp.hi{color:var(--tx);border-color:var(--tx3)}
.cal-day-hdr{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}

/* ── Tags and Method ── */
.dash-sec{display:flex;flex-direction:column;gap:12px}
#tagsBody{display:flex;flex-direction:column;gap:28px}
.learn-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.learn-card{background:var(--sf);border:1px solid var(--line);border-radius:10px;padding:16px}
.learn-card.lc-bull{border-left:3px solid var(--yes)}
.learn-card.lc-bear{border-left:3px solid var(--short)}
.learn-card-hdr{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.learn-card-hdr .bdg{margin:0}
.bb{color:var(--yes)}.br{color:var(--short)}
.learn-panels{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.lp-lbl{text-align:center;margin-bottom:4px}
.lp-svg{width:100%;display:block;background:var(--sf2);border-radius:5px}
.lp-line{fill:none;stroke:var(--tx3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.lp-zero{stroke:var(--line);stroke-width:0.5;stroke-dasharray:2 2}
.lp-dot-A{fill:var(--sf2);stroke:var(--tx2);stroke-width:1.5}
.lp-dot-B{fill:var(--sf2);stroke:var(--acc);stroke-width:1.5}
/* SVG user units: the 150-wide drawings render at 1.5x or more, so 8 is 12px on screen */
.lp-txt{font-family:var(--mono);font-size:8px;fill:var(--tx2);text-anchor:middle}
.lp-connect-bull{stroke:var(--yes);stroke-width:1.2;stroke-dasharray:3 2}
.lp-connect-bear{stroke:var(--short);stroke-width:1.2;stroke-dasharray:3 2}
.lp-cap{font-family:var(--sans);font-size:8px;text-anchor:middle}
.lp-cap-bull{fill:var(--yes)}
.lp-cap-bear{fill:var(--short)}
.rules{max-width:780px}
.rule{display:flex;gap:14px;padding:12px 0;border-bottom:1px solid var(--line)}
.rule:last-child{border-bottom:none}
.rule-n{font-family:var(--mono);font-size:12px;color:var(--tx3);flex-shrink:0;width:22px;padding-top:3px}
.rule-k{flex-shrink:0;width:170px}
.rule-b{font-size:14px;color:var(--tx2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<div class="shell">

<header class="topbar">
  <span class="t-logo">Divergence Screener</span>
  <nav class="nav">
    <button class="nav-btn active" data-pg="scan">Scan</button>
    <button class="nav-btn" data-pg="movers">Movers</button>
    <button class="nav-btn" data-pg="calendar">Calendar</button>
    <button class="nav-btn" data-pg="tags">Tags</button>
    <button class="nav-btn" data-pg="learn">Method</button>
  </nav>
  <div class="t-right">
    <div class="sdot" id="sdot"></div>
    <span class="slbl" id="slbl">Ready</span>
  </div>
</header>

<div id="pg-scan" class="page active">
  <section class="brief">
    <h1 id="briefH"></h1>
    <div class="state" id="briefState"></div>
    <div class="picks" id="briefPicks" style="display:none">
      <span class="lbl" id="briefLbl"></span>
      <div class="picks" id="picks"></div>
      <span class="calm">Passing the checklist is not a prediction: the backtest has not shown an edge yet, so size accordingly.</span>
    </div>
  </section>
  <section class="scan-sec">
    <div class="bar">
      <button class="rbtn" id="runBtn" onclick="startScan()">Run Scan</button>
      <span id="scan-age" class="audit"></span>
      <details class="set" id="scanSet"><summary>Scan settings</summary>
        <div class="setbody">
          <div class="ss"><div class="fl">Exchange</div>
            <select id="exchange"><option value="all" selected>All (Binance first)</option><option value="okx">OKX</option><option value="binance">Binance</option><option value="bybit">Bybit</option></select>
            <div class="hint">All: Binance data, OKX / Bybit for coins Binance does not list, and each setup checked on the other venues (ALSO ON).</div></div>

          <div class="ss"><div class="fl">Minimum weekly volume ($M)</div>
            <div class="nw"><button class="nb" onclick="stepVol(-10)" aria-label="Lower the volume floor">-</button><input type="number" id="minVol" value="__DEFAULT_VOL_M__" min="5" max="2000" step="10"><button class="nb" onclick="stepVol(10)" aria-label="Raise the volume floor">+</button></div>
            <div class="hint">Rolling 7-day traded value. $70M/week is about $10M/day.</div>
            <button class="gbtn" id="uniBtn" style="margin-top:8px" onclick="checkUniverse()">How many coins pass?</button>
            <div id="uniOut" class="hint"></div></div>

          <div class="ss"><div class="fl">Detection strictness</div>
            <div class="trow" id="strG">
              <button class="tbtn active" data-val="balanced">Balanced</button>
              <button class="tbtn" data-val="strict">Strict</button>
            </div>
            <div class="hint" id="strHint">Two MACD waves with a real bump between them; the price and MACD differences must clear a minimum size.</div></div>

          <div class="ss"><div class="fl">Max pairs to scan</div>
            <div class="nw"><button class="nb" onclick="stepPairs(-25)" aria-label="Fewer pairs">-</button><input type="number" id="maxPairs" value="__DEFAULT_MAX_PAIRS__" min="10" max="600" step="25"><button class="nb" onclick="stepPairs(25)" aria-label="More pairs">+</button></div>
            <div class="hint">Most liquid first. Raise it to reach deeper into the tail.</div></div>

          <div class="ss"><div class="fl">Scan speed</div>
            <div class="trow" id="spdG">
              <button class="tbtn" data-val="safe">Safe</button>
              <button class="tbtn active" data-val="normal">Normal</button>
              <button class="tbtn" data-val="fast">Fast</button>
            </div>
            <div class="hint">Fast uses more parallel requests. Drop to Safe if the exchange starts rate-limiting.</div></div>

          <div class="ss"><div class="fl">Specific coins (optional)</div>
            <input type="text" id="symOverride" placeholder="ENS, SOL, TAO">
            <div class="hint">Comma-separated. Overrides the volume and thin-coin filters entirely &mdash; scans exactly these.</div></div>
        </div>
      </details>
    </div>

    <div class="prog" id="pgSec" style="display:none">
      <div class="sl" id="sl">
        <div class="sbg"><div class="sbg-feed" id="sbgFeed"></div></div>
        <div class="lring">
          <svg viewBox="0 0 72 72"><circle class="rtrack" cx="36" cy="36" r="32"/><circle class="rfill" id="rfill" cx="36" cy="36" r="32"/></svg>
          <div class="rpct" id="rpct">0%</div>
        </div>
        <div class="lpair" id="lpair">Loading markets...</div>
        <div class="llbl">Scanning for divergences</div>
      </div>
      <div class="pb-bg"><div class="pb-fill" id="pbFill"></div></div>
      <div class="pb-lbl" id="pbLbl">-</div>
      <div class="lc" id="lc"></div>
    </div>

    <div id="resWrap" style="display:none" class="scan-sec">
      <div class="rbar">
        <span class="spill" id="stBull"><span class="num" id="cBull">0</span> long</span>
        <span class="spill" id="stBear"><span class="num" id="cBear">0</span> short</span>
        <span class="spill" id="stEnter"><span class="num" id="cEnter">0</span> Yes</span>
        <div class="fbar">
          <button class="ftab" data-f="live" onclick="setF(this)" title="Everything still worth watching: answer Yes or Not yet.">Live</button>
          <button class="ftab" data-f="enter" onclick="setF(this)" title="Only rows whose answer is Yes.">Enter now</button>
          <button class="ftab" data-f="bull" onclick="setF(this)" title="Bullish divergences (long side) answering Yes or Not yet.">Bullish</button>
          <button class="ftab" data-f="bear" onclick="setF(this)" title="Bearish divergences (short side) answering Yes or Not yet.">Bearish</button>
          <button class="ftab" data-f="both" onclick="setF(this)" title="The same side diverging on 4H and 1H at once.">4H+1H</button>
          <button class="ftab active" data-f="all" onclick="setF(this)" title="Every divergence found, including the No rows: ones that already played out, were stopped, or are too old. Until you pick this, those are folded under the table.">All</button>
        </div>
      </div>
      <div class="tw">
        <table id="resT">
          <thead><tr>
            <th title="The coin, then at most two of its tags and +N for the rest. Open the row for every tag; hover a tag for its definition, or open the Tags page for the full list.">Coin</th>
            <th title="long: a regular bullish divergence. Across two MACD troughs, price makes a lower low while the MACD line makes a higher trough. short: a regular bearish divergence, the mirror on two peaks. Then the timeframes that diverge; the brighter one supplies the answer and the levels.">Side and TF</th>
            <th title="Grade and quality score 0-100: A is 70 or more, B is 55-69, C is below 55. Pattern points (up to 70) plus context points (higher timeframe, OI with funding, BTC state: 10 each). Context the data cannot check is left out and the score is rescaled to 100 (tag PARTIAL DATA). It describes the setup; it is not a win rate. Run the backtest to see how each grade has done.">Grade</th>
            <th class="sortable sorted" data-sort="score" title="A checklist, not a prediction. First matching rule wins. No: the stop was hit, price already ran 1R, more than 6 completed 4H bars (24h) / 8 completed 1H bars (8h) have closed after the signal bar, or the grade is C. Not yet: counter trend below grade A, price more than 2% past the signal price, a USD CPI / PCE / PPI / jobs / FOMC / Fed Chair print due within 12h (4H) or 4h (1H) or printed in the last hour, or a signal bar still forming with no lower-timeframe trigger. Otherwise Yes. Under the answer: the first sentence of the reason; open the row for all of it. Default order: Yes, then Not yet, then No; highest score first, No rows newest first. Click to return to it.">Answer</th>
            <th title="The entry reference: the signal price (the close of the MACD cross that ends the second wave), or the lower-timeframe trigger price when that fired first. 1R / 2R / 3R are measured from it; open the row to see them.">Entry</th>
            <th title="Beyond price extreme B by 0.25 ATR, and the distance to it from the entry as a percentage. Price trading through it means the divergence has failed.">Stop</th>
            <th class="sortable" data-sort="oi" title="Change in this coin's open interest (in dollars) over the last 7 days. Context only: it never changes the answer or the score. n/a means the exchange has too little OI history for this contract. On OKX it covers all of the coin's contracts, not just this one. Click to sort, biggest build-up first.">7d OI</th>
          </tr></thead>
          <tbody id="resB"></tbody>
        </table>
        <button class="fold" id="fold" aria-expanded="false" style="display:none"></button>
      </div>
      <p class="note">Click a row for its chart, targets, tags and the full checklist. Each row shows two tags at most; the rest are inside.</p>
      <div id="scan-audit" class="audit"></div>
    </div>
  </section>
  <div class="cal-strip" id="dash-cal" style="display:none"></div>
</div>

<div id="pg-movers" class="page">
  <div class="sig-hdr">
    <div><div class="sig-ttl">Movers &mdash; daily watchlist</div>
    <div class="sig-sub" id="mv-sub">Unusual volume, unusual range, or a big day &mdash; measured on completed daily bars. Crypto perps only. Uses the volume floor, speed and max pairs from Scan settings on the Scan page.</div></div>
  </div>
  <div class="wbar">
    <select id="mvExchange"><option value="okx">OKX</option><option value="binance">Binance</option><option value="bybit">Bybit</option></select>
    <button class="rbtn" id="mvRunBtn" onclick="startMovers()">Scan Movers</button>
    <label class="mv-thin" title="Also measure coins trading $__MOVER_THIN_M__M a week up to the volume floor. They are listed in their own section and need a volume spike AND range expansion together, since on a thin coin one big order is a volume spike by itself. Expect worse fills, slippage through stops, and wicks that clear stops and snap back: size down."><input type="checkbox" id="mvThin" onchange="mvThinPref()"> Include thin coins ($__MOVER_THIN_M__M+/wk)</label>
    <span class="wprog" id="mvProg"></span>
  </div>
  <div class="sig-filters">
    <button class="sig-ftab active" data-mf="all" onclick="setMF(this)">All</button>
    <button class="sig-ftab" data-mf="top" onclick="setMF(this)">Top 4</button>
    <button class="sig-ftab" data-mf="vol" onclick="setMF(this)">Volume spike</button>
    <button class="sig-ftab" data-mf="rng" onclick="setMF(this)">Range expansion</button>
    <button class="sig-ftab" data-mf="hi" onclick="setMF(this)">At range high</button>
    <button class="sig-ftab" data-mf="lo" onclick="setMF(this)">At range low</button>
  </div>
  <div id="mv-audit" class="audit"></div>
  <div class="tw">
    <table id="mvT" style="display:none">
      <thead><tr>
        <th>#</th><th>Coin</th>
        <th title="How unusual the day was, 0-100: volume anomaly (35) + range expansion (30) + proximity to a 7-day boundary (20) + size of the move (15). Has no direction. Not a win rate, not backtested.">Standout</th>
        <th title="Last completed daily bar, close to close. Highlighted at __MOVER_CHG__%+, which is one of the three triggers. The small figure underneath is the live move since that close.">&Delta;1d</th>
        <th title="Traded value of the last completed day as a multiple of this coin's own 20-day average. __MOVER_VOL__x+ is a trigger. A self-ratio only, never a dollar figure.">Vol vs 20d</th>
        <th title="True range of the last completed day as a multiple of this coin's own 20-day ATR. __MOVER_RNG__x+ is a trigger.">Range vs ATR20</th>
        <th title="Where the live price sits between the 7-day low (0%) and the 7-day high (100%).">Position in 7d range</th>
        <th title="PDH / PDL: prior-day high and low. 7D: the 7-day range boundaries. The levels to plan around, and the point where you are wrong.">Levels</th>
        <th title="Why this coin is on the list. Any one trigger qualifies. Hover a tag for its definition and threshold.">Read</th>
      </tr></thead>
      <tbody id="mvB"></tbody>
    </table>
    <div class="no-sig" id="mvEmpty">No movers scan yet &mdash; hit Scan Movers.</div>
  </div>
</div>

<div id="pg-calendar" class="page">
  <div class="sig-hdr">
    <div><div class="sig-ttl">Calendar &mdash; scheduled market movers</div>
    <div class="sig-sub">US inflation, jobs, Fed and growth prints, plus the BoJ, ECB and BoE rate decisions. Everything else on the schedule is hidden unless you switch to All high-impact. Times are your local time; hover a chip for UTC.</div></div>
  </div>
  <div class="cal-next" id="cal-next"><span class="cal-next-meta">Loading calendar...</span></div>
  <div class="wbar">
    <div class="sig-filters">
      <button class="sig-ftab active" data-cf="movers" onclick="setCF(this)" title="Only the curated list: CPI, PCE, PPI, the jobs report, FOMC, Powell, GDP, retail sales, ISM, and the BoJ / ECB / BoE rate decisions.">Market movers</button>
      <button class="sig-ftab" data-cf="high" onclick="setCF(this)" title="Everything ForexFactory rates High impact, all currencies.">All high-impact</button>
    </div>
    <button class="rbtn" id="calRefreshBtn" onclick="refreshCalendar()">Refresh</button>
    <span class="wprog" id="cal-meta"></span>
    <a class="cal-out" style="margin-left:auto" href="https://www.forexfactory.com/calendar" target="_blank" rel="noopener"
       title="Open ForexFactory's calendar in a new tab. The feed carries no actual values, so this is where you check what a print came in at.">ForexFactory &#8599;</a>
  </div>
  <div id="cal-audit" class="audit"></div>
  <div class="cal-grid" id="cal-grid"></div>
  <div class="dash-sec">
    <div class="cal-day-hdr">
      <div class="dash-sec-ttl" id="cal-day-ttl">Select a day</div>
      <a class="cal-out" id="cal-ff-day" href="https://www.forexfactory.com/calendar" target="_blank" rel="noopener"
         title="Open this day on ForexFactory, where the actual values appear once a print lands.">Open this day on ForexFactory &#8599;</a>
    </div>
    <div class="tw">
      <table id="calT" style="display:none">
        <thead><tr>
          <th>Time</th><th>Cur</th><th>Event</th><th>Impact</th><th>Forecast</th><th>Previous</th><th>Actual</th>
        </tr></thead>
        <tbody id="calB"></tbody>
      </table>
      <div class="no-sig" id="calEmpty">Nothing scheduled.</div>
    </div>
  </div>
</div>

<div id="pg-tags" class="page">
  <div class="dash-sec">
    <div class="dash-sec-ttl">What every tag means</div>
    <div class="prose">
      Each tag as it appears on the Scan page, with the same wording as its hover text. A row shows
      two of its tags and +N; open the row for all of them, and for the regime, which sits in its
      Setup list. The regime tags set the grade the answer needs (COUNTER TREND needs A), TRIPLE and PARTIAL DATA change the
      score, and LTF / 15M TRIGGER allow Yes (early); every other tag, MACRO included, is context
      for you to weigh.
    </div>
  </div>
  <div id="tagsBody"></div>
</div>

<div id="pg-learn" class="page">
  <div class="dash-sec">
    <div class="dash-sec-ttl">What this screener looks for</div>
    <div class="prose">
      Regular divergence only &mdash; price and momentum disagreeing at a trend extreme, which is the
      classic <b>reversal</b> signature. Hidden (continuation) divergences are
      not scanned. It is read the way you would on a chart: the MACD/signal crossovers split the MACD line into waves, each wave below the signal line is one trough, and two neighbouring troughs A and B are compared with the price lows inside those same waves. Price lower, MACD trough higher: that mismatch is the whole signal.
    </div>
  </div>

  <div class="learn-grid">
    <div class="learn-card lc-bull">
      <div class="learn-card-hdr"><span class="bdg bb">REGULAR BULLISH</span><span class="learn-sub">Potential reversal UP</span></div>
      <div class="learn-panels">
        <div><div class="lp-lbl">Price</div>
          <svg viewBox="0 0 150 82" class="lp-svg">
            <polyline points="8,18 22,28 32,22 45,37 60,28 78,48 92,40 110,60 126,52 142,68" class="lp-line"/>
            <circle cx="45" cy="37" r="4" class="lp-dot-A"/><text x="45" y="27" class="lp-txt">A</text>
            <circle cx="110" cy="60" r="4" class="lp-dot-B"/><text x="110" y="50" class="lp-txt">B</text>
            <line x1="45" y1="37" x2="110" y2="60" class="lp-connect-bear"/>
            <text x="77" y="79" class="lp-cap lp-cap-bear">Lower low</text>
          </svg></div>
        <div><div class="lp-lbl">MACD</div>
          <svg viewBox="0 0 150 82" class="lp-svg">
            <line x1="5" y1="42" x2="145" y2="42" class="lp-zero"/>
            <polyline points="8,50 22,60 32,52 45,68 60,60 78,54 92,50 110,50 126,46 142,40" class="lp-line"/>
            <circle cx="45" cy="68" r="4" class="lp-dot-A"/><text x="36" y="79" class="lp-txt">A</text>
            <circle cx="110" cy="50" r="4" class="lp-dot-B"/><text x="119" y="42" class="lp-txt">B</text>
            <line x1="45" y1="68" x2="110" y2="50" class="lp-connect-bull"/>
            <text x="77" y="13" class="lp-cap lp-cap-bull">Higher low</text>
          </svg></div>
      </div>
    </div>

    <div class="learn-card lc-bear">
      <div class="learn-card-hdr"><span class="bdg br">REGULAR BEARISH</span><span class="learn-sub">Potential reversal DOWN</span></div>
      <div class="learn-panels">
        <div><div class="lp-lbl">Price</div>
          <svg viewBox="0 0 150 82" class="lp-svg">
            <polyline points="8,68 22,58 32,64 45,52 58,62 76,44 90,50 110,30 126,42 142,20" class="lp-line"/>
            <circle cx="45" cy="52" r="4" class="lp-dot-A"/><text x="45" y="44" class="lp-txt">A</text>
            <circle cx="110" cy="30" r="4" class="lp-dot-B"/><text x="110" y="22" class="lp-txt">B</text>
            <line x1="45" y1="52" x2="110" y2="30" class="lp-connect-bull"/>
            <text x="77" y="13" class="lp-cap lp-cap-bull">Higher high</text>
          </svg></div>
        <div><div class="lp-lbl">MACD</div>
          <svg viewBox="0 0 150 82" class="lp-svg">
            <line x1="5" y1="42" x2="145" y2="42" class="lp-zero"/>
            <polyline points="8,48 22,38 32,44 45,30 60,38 78,36 92,42 110,48 126,44 142,50" class="lp-line"/>
            <circle cx="45" cy="30" r="4" class="lp-dot-A"/><text x="45" y="22" class="lp-txt">A</text>
            <circle cx="110" cy="48" r="4" class="lp-dot-B"/><text x="119" y="60" class="lp-txt">B</text>
            <line x1="45" y1="30" x2="110" y2="48" class="lp-connect-bear"/>
            <text x="77" y="79" class="lp-cap lp-cap-bear">Lower high</text>
          </svg></div>
      </div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">The rules a signal has to pass (Balanced)</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Completed candles only.</b> The still-forming bar is dropped before anything is calculated, so two scans inside the same bar agree. The one exception is a row whose signal bar is still forming (its answer and its opened row say so): it is provisional and can disappear.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Waves come from the crossovers.</b> A bear wave runs from a bearish MACD/signal cross to the next bullish cross, a bull wave the other way. A cross only counts if the two lines separate by 5% of the recent MACD range within 3 bars, and a wave shorter than 3 bars is merged into its neighbours, so a tiny double dot cannot split one trough into two.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>One trough per wave.</b> The MACD trough is the lowest MACD-line value in the wave. The price low is the lowest wick from the start of the wave to 2 bars after it ends, because price often bottoms just after momentum turns.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>A then B, with a real bump between.</b> Two neighbouring troughs, 8 to 80 bars apart. The MACD peak between them must rise at least 25% of the leg's MACD range above the higher trough: that is the middle of the W.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Minimum size.</b> The price difference must clear both a fixed floor and a fraction of ATR, and the MACD difference must be at least 15% of the leg's MACD range. This is what kills micro-divergences.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>Reported only when wave B is closed.</b> The cross that ends wave B is the signal bar and its close is the signal price. Before that cross there is no signal.</div></div>
      <div class="rule"><div class="rule-n">07</div><div class="rule-b"><b>Liquidity gate.</b> Rolling 7-day traded value must clear the floor you set. Measured from the candles as a ratio against the exchange's own 24h figure, so it does not depend on guessing whether a venue reports volume in coins, contracts or dollars. Thin coins are left out too: a top-of-book spread above 0.1%, a 4H ATR outside 0.6% to 8% of price, or (for a coin with a setup) open interest under $5M. On OKX that open interest is the whole coin's, across all its contracts. A timeframe with fewer than 300 completed bars is not analysed, so the 200 EMA has settled. Unknown spread or OI never excludes a coin, and a symbol override skips all of these. The audit line under the table counts each.</div></div>
      <div class="rule"><div class="rule-n">08</div><div class="rule-b"><b>One row per coin, Binance first.</b> The All exchange setting reads Binance; a coin Binance does not list is read on whichever of OKX / Bybit trades more of it and tagged NOT ON BINANCE. For each row the same detection then runs on the other venues listing the coin, and the row is tagged ALSO ON with those where the same side fires within one bar. A tag, never points. Picking one exchange scans only that one, with no venue tags.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">The answer &mdash; a checklist, not a prediction</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>No comes first.</b> The stop was hit, price already made 1R or more, more than 6 completed 4H bars (24h) / 8 completed 1H bars (8h) have closed after the signal bar (counted to the last completed bar, not the clock), or the grade is C.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Not yet names the one thing missing.</b> A counter-trend setup below grade A (watch only), a pullback to the signal price after price ran more than 2%, a high-impact USD print (CPI, Core PCE, PPI, non-farm payrolls, unemployment rate, FOMC rate / statement / press conference, Fed Chair speech or testimony) due within 12 hours for a 4H signal or 4 hours for a 1H signal, or printed in the last hour, or a signal bar that has not closed and has no lower-timeframe trigger. BoJ / ECB / BoE rate decisions and the rest of the market movers never block; one due within 24 hours shows as a MACRO tag, and so does a USD print above that the answer does not name (e.g. CPI in 6 hours on a 1H row).</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Yes (confirmed) and Yes (early).</b> Confirmed means the signal bar has closed and nothing above applies. Early means the signal bar is still forming but the lower timeframe (1H or 15m for 4H, 15m for 1H) has already crossed its MACD and closed beyond its last swing since B. A 4H signal's 15m trigger counts only if it closes within 8 hours of the 4H close.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Regime changes the bar, it never blocks.</b> BTC 4H is BULL (RSI above 55 and close above its 200 EMA), BEAR (RSI below 45 and close below it) or CHOP (anything else). WITH TREND means the BTC state and the coin's own 200 EMA both agree with the signal. CHOP keeps the normal cutoff, because ranges are where reversals work. Anything else, missing BTC data included, is COUNTER TREND and needs grade A.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>The answer ages.</b> It uses the live price at the moment of the scan. Once the scan is more than an hour old the answers dim and the page asks for a re-run. The brief on top of the Scan page is built from the same rows: BTC's state and the nearest USD print come from the scan, and only the print's countdown keeps running.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>One row per coin and side.</b> When 4H and 1H diverge the same way the row shows both timeframes and takes its answer, grade and levels from the better one (Yes beats Not yet beats No, 4H wins a tie); the brighter one shows which. Open a row for each timeframe's levels and its chart.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Context tags &mdash; what else is true about this coin</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Movers and OI-flow tags never change the answer.</b> They are fetched only for coins that already have a divergence, and they are yours to weigh. The grade does use some of the same data: the higher timeframe's MACD and the change in open interest into B are score components.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Movers tags describe the last completed day.</b> VOLUME SPIKE and RANGE EXPANSION are measured against the coin's own 20-day baseline, BIG MOVE is a 5%+ day, and the range tags say price is at or through its 7-day high or low. A bearish divergence AT RANGE HIGH is the textbook fade; the same divergence mid-range is weaker.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>OI tags describe 7-day positioning.</b> Price and open interest together give the quadrant: NEW LONGS, NEW SHORTS, SHORT COVERING, CAPITULATION. Funding refines it: SPOT-LED is the healthy version of new longs, CROWDED LONGS is leverage arriving late, SQUEEZE FUEL is shorts crowded and paying. No OI tag means open interest moved less than 5% or the contract is too new to judge.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>HIGH OI/VOL is a crowding warning, not a direction.</b> Open interest above about 1.5x average daily volume means positions cannot exit without moving the price. Size down.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Movers &mdash; what a daily watchlist is actually screening for</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Three triggers, any one qualifies.</b> A 1-day move of 5%+, <i>or</i> the day's traded value at 2x its own 20-day average, <i>or</i> the day's true range at 1.5x ATR20. A coin only has to be unusual in one way to be worth a look.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Volume and range beat raw % change.</b> Sorting the market by 24h gainers is the lazy version of this screen: it surfaces whatever already moved and misses a dormant coin waking up on 3x volume. Both ratios are measured against each coin's <b>own</b> 20-day baseline, so a quiet large-cap and a volatile micro-cap are judged on the same scale.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Volume is only ever a self-ratio.</b> The OHLCV volume column means different things per exchange &mdash; coins, contracts, or quote currency. Comparing one coin's day against its own average divides that unknown factor out; comparing dollar amounts across venues would not.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Every row arrives with levels.</b> Prior-day high and low, the 7-day range boundaries, and where price sits inside that range right now. A watchlist without a level and a point where you are wrong is a list of names, not a plan &mdash; and that is the part these posts usually leave out.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Completed bars only.</b> The scan describes the last <b>closed</b> day, so it reads the same at 02:00 and at 22:00. It tells you what happened yesterday so you can plan today; it is not a live intraday tape.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>No stocks, ETFs or commodities.</b> Those perps are excluded from the universe entirely. Their underlying market closes at the weekend, leaving a thin book, funding-driven drift and a gap at the Monday cash open &mdash; quoting a weekend level on one is quoting a level on nothing.</div></div>
      <div class="rule"><div class="rule-n">07</div><div class="rule-b"><b>The standout score is a description, not an edge.</b> Volume anomaly (35), range expansion (30), proximity to a 7-day boundary (20), size of the move (15). It has no direction and has not been backtested &mdash; it is not a win rate.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Quality score and grade</div>
    <div class="prose">
      0 to 100, graded <b>A</b> (70+), <b>B</b> (55-69) or
      <b>C</b> (below 55). Pattern points, up to 70: the price difference A to B in ATR
      (12), the MACD trough improvement (12), the height of the bump (8), both troughs on the classic side of
      zero (10), volume drying up in wave B (10), a rejection wick at B (10), a triple divergence (8). Context
      points, 10 each: the higher timeframe's MACD agreeing, open interest falling into B with funding not
      against it, and the BTC 4H state on the signal's side (BULL for bullish, BEAR for bearish; CHOP earns
      nothing). Open interest rising more than 5% against the signal costs 10.
      <br><br>
      A context item the data cannot check (no higher-timeframe candles, no OI history or funding, no BTC
      candles) is left out rather than scored as a miss: the score is what was earned out of what could be
      checked, scaled to 100, and the signal is tagged <b>PARTIAL DATA</b>.
      <br><br>
      The grade describes the setup. It is not a win rate. To see how each grade and regime has actually done on
      the stored candles, run <b>python macd_divergence_screener_v5.py --backtest</b>.
    </div>
  </div>
</div>

</div>

<script>
var TFM = __TF_CFG_JSON__;

var SC = {active:'at the signal', extended:'extended', ran:'ran', invalidated:'invalidated'};

function qcell(q){
  return '<div class="qc"><span class="num">' + q + ' <span class="lo">' + (q >= 70 ? 'HIGH' : q >= 50 ? 'MED' : 'LOW') +
    '</span></span><div class="qbar"><div class="qfill" style="width:' + q + '%"></div></div></div>';
}
function gradeCell(p){ return '<span class="num">' + p.grade + ' ' + p.score + '</span>'; }
// Sub-cent tokens round to values like 9.8e-06, and JS stringifies anything
// below 1e-6 with an exponent — an unusable price level. Force plain digits.
function fmtPx(v){
  if(v === null || v === undefined) return '-';
  if(v !== 0 && Math.abs(v) < 1e-4) return v.toFixed(12).replace(/0+$/, '');
  return String(v);
}
function fmtVol(v){
  if(v == null) return '&mdash;';
  if(v >= 1e9) return '$' + (v/1e9).toFixed(2) + 'B';
  if(v >= 1e6) return '$' + (v/1e6).toFixed(0) + 'M';
  return '$' + Math.round(v).toLocaleString();
}
function b2t(bars, tf){
  var m = bars * ((TFM[tf] && TFM[tf].minutes) || 240);
  if(m < 60) return m + 'm ago';
  if(m < 1440) return Math.round(m/60) + 'h ago';
  var d = Math.round(m/1440);
  return d + (d === 1 ? ' day' : ' days') + ' ago';
}

var ANS = {enter:'Yes', wait:'Not yet', skip:'No'};
var EV_RANK = {enter:0, wait:1, skip:2};
function verdict(r){ return (r.entry && r.entry.verdict) || 'skip'; }
function primaryTf(r){ return r.primary_tf || r.tfs[0]; }
function primary(r){ return r.signals[primaryTf(r)]; }
function rowKey(r){ return r.pair + '|' + r.side; }

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
/* The full reason. It can carry a calendar event title from an outside feed. */
function reason(r){
  var e = r.entry || {};
  return calEsc(e.reason) + (e.level != null ? ' ' + fmtPx(e.level) : '');
}
/* Its first sentence, without the leading Yes / Not yet / No the pill already says. */
function shortWhy(r){ return reason(r).replace(/^(No|Not yet|Yes[^.]*)\.\s*/, '').split('. ')[0].replace(/\.$/, ''); }
function ansPill(r){ var v = verdict(r); return '<span class="ans ' + v + '">' + ANS[v] + '</span>'; }
function evCell(r){ return ansPill(r) + '<span class="sub">' + shortWhy(r) + '</span>'; }
function sideHtml(r){
  var s = r.side === 'bullish' ? 'long' : 'short';
  return '<span class="side-' + s + '">' + s + '</span>' + r.tfs.map(function(t){
    return '<span class="tf' + (t === primaryTf(r) ? ' on' : '') + '">' + tfLabel(t) + '</span>'; }).join('');
}
function tag(k, txt){ return '<span class="tag"' + tip(k) + '>' + (txt || k) + '</span>'; }
/* Every tag of a row, pattern tags first. The Coin cell shows two; the opened row shows all. */
function tagChips(r){
  var h = (primary(r).tags || []).map(function(f){ return tag(f); });
  (r.venue_tags || []).forEach(function(f){ h.push(tag(f.indexOf('ALSO ON') === 0 ? 'ALSO ON' : f, f)); });
  if(r.macro) h.push(tag('MACRO', calEsc(r.macro)));
  ((r.mover || {}).flags || []).forEach(function(f){ h.push(tag(f)); });
  if(r.flow && r.flow.quadrant){
    h.push(tag(r.flow.quadrant));
    (r.flow.flags || []).forEach(function(f){ h.push(tag(f)); });
  }
  return h;
}
function tfLabel(tf){ return (TFM[tf] && TFM[tf].label) || tf.toUpperCase(); }

/* Every coin the scan did not actually analyse gets named here. The point is
   that "3 live setups" must never be readable as "the market is quiet" when
   40 coins silently failed to fetch or the scan was cut off half way. */
function auditLine(m){
  if(!m || !m.dispatched) return '';
  var bits = [];
  bits.push(m.scanned + ' of ' + m.dispatched + ' candidate pairs analysed');
  if(m.vol_rejected) bits.push(m.vol_rejected + ' below the 7-day volume floor');
  if(m.spread_rejected) bits.push(m.spread_rejected + ' spread above 0.1%');
  if(m.atr_rejected)    bits.push(m.atr_rejected + ' 4H ATR outside 0.6-8% of price');
  if(m.short_history)   bits.push(m.short_history + ' timeframe(s) skipped: under 300 completed bars');
  if(m.oi_rejected)     bits.push(m.oi_rejected + ' with a setup dropped: open interest under $5M');
  if(m.no_data)      bits.push(m.no_data + ' returned no usable candles');
  if(m.no_1h)        bits.push(m.no_1h + ' analysed on 4H only (no usable 1H candles)');
  if(m.errors)       bits.push(m.errors + ' errored');
  if(m.no_ticker)    bits.push(m.no_ticker + ' had no 24h volume to pre-filter on');
  var txt = bits.join(' · ');
  if(m.complete === false){
    return '<div class="audit warn">&#9888; This scan did not finish - it was interrupted, so the ' +
           'coins after the cut-off were never looked at. Re-run before trusting an empty result.<br>' + txt + '</div>';
  }
  return txt;
}

function showPg(name){
  document.querySelectorAll('.page').forEach(function(p){ p.classList.remove('active'); });
  document.getElementById('pg-' + name).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(function(b){ b.classList.toggle('active', b.dataset.pg === name); });
  if(name === 'movers') loadMovers();
  if(name === 'calendar') loadCalendar();
}
document.querySelectorAll('.nav-btn').forEach(function(b){
  b.addEventListener('click', function(){ showPg(b.dataset.pg); });
});

var TH = {chg:__MOVER_CHG__, vol:__MOVER_VOL__, rng:__MOVER_RNG__, edge:__MOVER_EDGE__, thinM:__MOVER_THIN_M__,
          oiVol:__OI_VOL__, fundSpot:__FUND_SPOT__, fundCrowd:__FUND_CROWD__, fundSqueeze:__FUND_SQUEEZE__};
var TIP = {
  'NEW LONGS':       'Price up + OI up: new money opening longs. A real trend if funding is not rich.',
  'NEW SHORTS':      'Price down + OI up: new money opening shorts. A real downtrend, or a squeeze being loaded.',
  'SHORT COVERING':  'Price up + OI down: shorts closing, not buyers arriving. Movement without new conviction.',
  'CAPITULATION':    'Price down + OI down: longs closing. Movement without new conviction.',
  'SPOT-LED':        'New longs with funding at or below ' + TH.fundSpot + '%/day: buyers are in spot while shorts fight it. The healthy version.',
  'CROWDED LONGS':   'New longs with funding at or above ' + TH.fundCrowd + '%/day: leverage arriving late and paying for it.',
  'SQUEEZE FUEL':    'New shorts with funding at or below ' + TH.fundSqueeze + '%/day: shorts are crowded and paying. Fuel for a squeeze.',
  'HIGH OI/VOL':     'Open interest above ' + TH.oiVol + 'x average daily volume: positions cannot exit without moving price. A crowding warning, no direction.',
  'VOLUME SPIKE':    'Traded value at ' + TH.vol + 'x+ this coin\'s own 20-day average. Trigger.',
  'RANGE EXPANSION': 'True range at ' + TH.rng + 'x+ this coin\'s own 20-day ATR. Trigger.',
  'BIG MOVE':        'Close-to-close move of ' + TH.chg + '%+ on the last completed day. Trigger.',
  'AT RANGE HIGH':   'Within ' + TH.edge + '% of the 7-day high.',
  'ABOVE 7D HIGH':   'Price has traded through the 7-day high.',
  'AT RANGE LOW':    'Within ' + TH.edge + '% of the 7-day low.',
  'BELOW 7D LOW':    'Price has traded through the 7-day low.',
  'TRIPLE':          'The trough before A also diverges from A: three troughs in a row, price lower each time, MACD higher each time.',
  'W-PATTERN':       'The bump between A and B is at least half the MACD range of the leg: a clear W (M for bearish).',
  'LTF TRIGGER':     'The lower timeframe (1H for a 4H signal, 15m for a 1H signal) has crossed its MACD and closed beyond its last swing since B.',
  '15M TRIGGER':     '4H signals only: the 15m chart has crossed its MACD and closed beyond its last swing since B, within 8 hours of the 4H close. Your actual entry moment; the trigger price is whichever trigger (1H or 15m) closed first.',
  'MACRO':           'A calendar print due within 24 hours that this row\'s answer does not name: a curated market mover that never blocks (BoJ / ECB / BoE rates, GDP, retail sales, ISM, FOMC minutes, ...), or a USD print that can block (CPI, PCE, PPI, jobs, FOMC rate, Fed Chair) but is outside this timeframe\'s wait window (12h on 4H, 4h on 1H) or the answer is already No / Not yet for another reason. The sooner one is shown. Context only.',
  '7d OI':           'Change in the coin\'s open interest over the last 7 days. Rising = positions being opened, falling = positions being closed. Read it with price: the four positioning tags below are exactly that combination. n/a = too little OI history. On OKX it covers all of the coin\'s contracts, not just this one.',
  'ZERO-LINE CROSS': 'The MACD line has already crossed zero since the signal bar.',
  'WITH TREND':      'The BTC 4H state agrees with the signal (BULL: RSI above 55 and close above its 200 EMA; BEAR: RSI below 45 and close below it) and so does the coin\'s own 200 EMA. Grade B is enough for Yes.',
  'COUNTER TREND':   'BTC (BULL or BEAR) or the coin\'s 200 EMA disagrees with the signal, or BTC data is missing. Needs grade A for Yes; otherwise watch only.',
  'CHOP':            'BTC 4H is neither BULL nor BEAR: RSI between 45 and 55, or RSI and the 200 EMA disagree. A reversal keeps the normal cutoff (grade B is enough for Yes); the BTC score points are not earned.',
  'PARTIAL DATA':    'Some score context could not be checked (higher timeframe, open interest or funding, or BTC). It is left out and the score is what was earned out of what could be checked, scaled to 100.',
  'NOT ON BINANCE':  'All-exchanges scan only. Binance does not list this coin, so its data comes from whichever of OKX / Bybit trades more of it (the venue is Data from in the opened row).',
  'ALSO ON':         'All-exchanges scan only. The same divergence, same side and timeframe, also fires on the named venue(s) within one bar of this one. Confirmation only: it adds no points. A venue that could not be fetched is simply not named.'
};
function tip(k){ return TIP[k] ? ' title="' + TIP[k] + '"' : ''; }

/* ── Tags page: built from TIP so it cannot drift from the hover text ─────── */
var TAG_GROUPS = [
  ['Pattern tags (about the divergence itself)', ['TRIPLE','W-PATTERN','LTF TRIGGER','15M TRIGGER','ZERO-LINE CROSS']],
  ['Regime (sets how high the grade must be)', ['WITH TREND','CHOP','COUNTER TREND']],
  ['Score', ['PARTIAL DATA']],
  ['Exchanges (All scan only, e.g. ALSO ON OKX, BYBIT)', ['NOT ON BINANCE','ALSO ON']],
  ['Calendar (shown with the print and its countdown, e.g. MACRO BOJ Policy Rate in 6h)', ['MACRO']],
  ['Open interest: 7-day positioning', ['7d OI','NEW LONGS','NEW SHORTS','SHORT COVERING','CAPITULATION']],
  ['Open interest: funding and crowding', ['SPOT-LED','CROWDED LONGS','SQUEEZE FUEL','HIGH OI/VOL']],
  ['Daily movers (the last completed day)', ['VOLUME SPIKE','RANGE EXPANSION','BIG MOVE','AT RANGE HIGH','ABOVE 7D HIGH','AT RANGE LOW','BELOW 7D LOW']]
];
document.getElementById('tagsBody').innerHTML = TAG_GROUPS.map(function(g){
  return '<div class="dash-sec"><div class="dash-sec-ttl">' + g[0] + '</div><div class="rules">' +
    g[1].map(function(k){
      return '<div class="rule"><div class="rule-k">' + tag(k) + '</div><div class="rule-b">' + TIP[k] + '</div></div>';
    }).join('') + '</div></div>';
}).join('');
function oiVal(r){ return r.flow && r.flow.oi_chg_pct != null ? r.flow.oi_chg_pct : null; }
function oiCell(r){
  var v = oiVal(r);
  return v == null ? 'n/a' : (v > 0 ? '+' : '') + v + '%';
}

function fmtPct(v){
  if(v === null || v === undefined) return '&mdash;';
  return (v >= 0 ? '+' : '') + v + '%';
}
/* ── Movers (daily watchlist) ─────────────────────────────────────────────── */
var _mf = 'all', _mrows = [], mes = null;

function mMatches(r, f){
  var fl = r.flags || [];
  if(f === 'vol') return fl.indexOf('VOLUME SPIKE') >= 0;
  if(f === 'rng') return fl.indexOf('RANGE EXPANSION') >= 0;
  if(f === 'hi')  return fl.indexOf('AT RANGE HIGH') >= 0 || fl.indexOf('ABOVE 7D HIGH') >= 0;
  if(f === 'lo')  return fl.indexOf('AT RANGE LOW') >= 0 || fl.indexOf('BELOW 7D LOW') >= 0;
  return true;
}
function setMF(btn){
  document.querySelectorAll('.sig-ftab[data-mf]').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active'); _mf = btn.dataset.mf; renderMovers();
}
function fmtX(v){ return (v === null || v === undefined) ? '&mdash;' : v.toFixed(2) + 'x'; }

/* Same contract as the other two audit lines: every pair the scan did not
   actually measure gets named, so a short list can never be misread as a
   quiet market. */
function moversAudit(m){
  if(!m || !m.dispatched) return '';
  var bits = [m.analysed + ' of ' + m.dispatched + ' pairs measured',
              (m.quiet || 0) + ' unremarkable (no trigger hit)'];
  if(m.vol_rejected)    bits.push(m.vol_rejected + ' below the volume floor');
  if(m.no_data)         bits.push(m.no_data + ' had too little daily history');
  if(m.errors)          bits.push(m.errors + ' errored');
  if(m.tradfi_excluded) bits.push(m.tradfi_excluded + ' stock/ETF/commodity perp(s) excluded by design');
  if(m.thin) bits.push((m.thin_analysed || 0) + ' thin pair(s) measured, ' + (m.thin_quiet || 0) +
                       ' without volume spike + range expansion together');
  var txt = bits.join(' · ');
  if(m.complete === false){
    return '<div class="audit warn">&#9888; This movers scan did not finish &mdash; re-run before trusting it.<br>' + txt + '</div>';
  }
  return txt;
}
function mvRow(r, i, thin){
  var flags = (r.flags || []).map(function(f){ return tag(f); }).join('') || '&mdash;';
  var hl = function(on){ return on ? 'hl' : 'lo'; };
  var badge = thin ? '<span class="thin-badge" title="Below the volume floor. Thin book: expect worse fills and slippage through stops. Size down.">THIN &middot; ' +
                     fmtVol(r.daily_volume) + '/day</span>' : '';
  var tr = document.createElement('tr');
  tr.innerHTML =
    '<td><span class="mrank' + (!thin && i < 4 ? ' top' : '') + '">' + (i + 1) + '</span></td>' +
    '<td><b>' + r.base + '</b><span class="sub num">' + fmtPx(r.current_price) + '</span>' + badge + '</td>' +
    '<td>' + qcell(r.score) + '</td>' +
    '<td class="num"><span class="' + hl(Math.abs(r.chg_1d_pct || 0) >= TH.chg) + '">' + fmtPct(r.chg_1d_pct) + '</span>' +
      '<span class="sub">today ' + fmtPct(r.chg_live_pct) + '</span></td>' +
    '<td class="num ' + hl((r.vol_mult || 0) >= TH.vol) + '">' + fmtX(r.vol_mult) + '</td>' +
    '<td class="num ' + hl((r.range_mult || 0) >= TH.rng) + '">' + fmtX(r.range_mult) + '</td>' +
    '<td><div class="mpos"><i style="left:' + r.pos_in_range + '%"></i></div>' +
      '<span class="sub num">' + r.pos_in_range + '% of range</span></td>' +
    '<td><div class="mlv"><small>PDH</small> ' + fmtPx(r.pdh) + '</div><div class="mlv"><small>PDL</small> ' + fmtPx(r.pdl) + '</div>' +
      '<div class="mlv"><small>7D</small> ' + fmtPx(r.r_lo) + ' &ndash; ' + fmtPx(r.r_hi) + '</div></td>' +
    '<td>' + flags + '</td>';
  return tr;
}
/* Thin rows never share a ranking with the main list: a $3M/day coin with a
   whale print would otherwise top it on score alone. Top 4 is main-list only. */
function renderMovers(){
  var rows = _mrows.slice().sort(function(a, b){ return b.score - a.score; })
                   .filter(function(r){ return mMatches(r, _mf); });
  var core = rows.filter(function(r){ return r.tier !== 'thin'; });
  var thin = _mf === 'top' ? [] : rows.filter(function(r){ return r.tier === 'thin'; });
  if(_mf === 'top') core = core.slice(0, 4);
  var tb = document.getElementById('mvB'); tb.innerHTML = '';
  if(core.length + thin.length === 0){
    document.getElementById('mvT').style.display = 'none';
    var e = document.getElementById('mvEmpty');
    e.style.display = '';
    e.textContent = _mrows.length ? 'No pairs match this filter.' : 'No movers scan yet - hit Scan Movers.';
    return;
  }
  document.getElementById('mvEmpty').style.display = 'none';
  document.getElementById('mvT').style.display = 'table';
  core.forEach(function(r, i){ tb.appendChild(mvRow(r, i, false)); });
  if(thin.length){
    var sep = document.createElement('tr');
    sep.className = 'mv-sep';
    sep.innerHTML = '<td colspan="9">&#9888; LOW LIQUIDITY &mdash; $' + TH.thinM + 'M/week up to the floor &middot; ' +
      'listed only on volume spike AND range expansion together &middot; thin book, wider fills, size down</td>';
    tb.appendChild(sep);
    thin.forEach(function(r, i){ tb.appendChild(mvRow(r, i, true)); });
  }
}
function mvThinPref(){
  try { localStorage.setItem('mvThin', document.getElementById('mvThin').checked ? '1' : '0'); } catch(x){}
}
function loadMovers(){
  fetch('/mover_signals').then(function(r){ return r.json(); }).then(function(d){
    _mrows = d.rows || [];
    var m = d.meta || {};
    if(d.ts){
      var ma = Math.round((Date.now() - new Date(d.ts)) / 60000);
      document.getElementById('mv-sub').textContent =
        _mrows.length + ' movers on ' + (m.exchange || '').toUpperCase() +
        ' · last completed daily bar vs its ' + (m.lookback_d || 20) +
        '-day baseline · scanned ' + ma + ' min ago';
      if(m.exchange) document.getElementById('mvExchange').value = m.exchange;
    }
    var pref = null;
    try { pref = localStorage.getItem('mvThin'); } catch(x){}
    document.getElementById('mvThin').checked = pref !== null ? pref === '1' : !!m.thin;
    document.getElementById('mv-audit').innerHTML = moversAudit(m);
    renderMovers();
  }).catch(function(){});
}
function startMovers(){
  if(mes){ mes.close(); mes = null; }
  var ex  = document.getElementById('mvExchange').value;
  var vol = (parseInt(document.getElementById('minVol').value) || 70) * 1e6;
  var spd = document.querySelector('#spdG .tbtn.active').dataset.val;
  var mp  = parseInt(document.getElementById('maxPairs').value) || 400;
  var btn = document.getElementById('mvRunBtn');
  btn.disabled = true;
  _mrows = [];
  renderMovers();
  document.getElementById('mv-audit').innerHTML = '';
  var mtot = 0, prog = document.getElementById('mvProg');
  prog.textContent = 'Loading markets...';

  var thin = document.getElementById('mvThin').checked ? '1' : '0';
  mes = new EventSource('/movers?exchange=' + ex + '&min_weekly_vol=' + vol +
                        '&speed=' + spd + '&max_pairs=' + mp + '&thin=' + thin);
  mes.addEventListener('log', function(e){ prog.textContent = JSON.parse(e.data).msg; });
  mes.addEventListener('total', function(e){ mtot = JSON.parse(e.data).total; });
  mes.addEventListener('progress', function(e){
    var d = JSON.parse(e.data);
    prog.textContent = d.current + '/' + mtot + ' - ' + d.pair.replace('/USDT:USDT','');
  });
  mes.addEventListener('result', function(e){
    _mrows.push(JSON.parse(e.data));
    renderMovers();
  });
  mes.addEventListener('done', function(e){
    var d = JSON.parse(e.data);
    mes.close(); btn.disabled = false;
    prog.textContent = 'Done - ' + d.hits + ' mover(s)';
    var m = d.meta || {};
    document.getElementById('mv-audit').innerHTML = moversAudit(m);
    document.getElementById('mv-sub').textContent =
      d.hits + ' movers on ' + (m.exchange || '').toUpperCase() +
      ' · last completed daily bar vs its ' + (m.lookback_d || 20) + '-day baseline · just now';
  });
  mes.addEventListener('error', function(e){
    if(!e.data) return;
    var msg = 'Scan error';
    try { msg = JSON.parse(e.data).msg; } catch(x){}
    prog.textContent = msg; mes.close(); btn.disabled = false;
  });
  mes.onerror = function(){
    if(mes.readyState === EventSource.CLOSED) return;
    // Same reasoning as the other two scans: an EventSource reconnect silently
    // launches a SECOND full scan against the same exchange quota. Close it and
    // let the user decide to re-run.
    mes.close(); btn.disabled = false;
    prog.textContent = 'Connection lost - scan stopped.';
  };
}

/* ── Scan settings ─────────────────────────────────────────────────────── */
var STR_HINTS = {
  balanced: 'Two MACD waves with a real bump between them; the price and MACD differences must clear a minimum size.',
  strict:   'Larger minimum size, and both troughs must sit below the zero line (both peaks above it for bearish).'
};
['strG','spdG'].forEach(function(g){
  document.querySelectorAll('#' + g + ' .tbtn').forEach(function(b){
    b.addEventListener('click', function(){
      document.querySelectorAll('#' + g + ' .tbtn').forEach(function(x){ x.classList.remove('active'); });
      b.classList.add('active');
      if(g === 'strG') document.getElementById('strHint').textContent = STR_HINTS[b.dataset.val];
    });
  });
});
function stepVol(d){
  var i = document.getElementById('minVol');
  i.value = Math.max(5, Math.min(2000, (parseInt(i.value) || 70) + d));
}
function stepPairs(d){
  var i = document.getElementById('maxPairs');
  i.value = Math.max(10, Math.min(600, (parseInt(i.value) || 400) + d));
}
function checkUniverse(){
  var btn = document.getElementById('uniBtn'), out = document.getElementById('uniOut');
  var ex = document.getElementById('exchange').value;
  var vol = (parseInt(document.getElementById('minVol').value) || 70) * 1e6;
  btn.disabled = true; btn.textContent = 'Checking...';
  out.textContent = '';
  fetch('/universe?exchange=' + ex + '&min_weekly_vol=' + vol)
    .then(function(r){ return r.json(); })
    .then(function(d){
      btn.disabled = false; btn.textContent = 'How many coins pass?';
      if(d.error){ out.textContent = 'Error: ' + d.error; return; }
      out.innerHTML = '<b>' + d.passing + '</b> of ' + d.total_perps +
        ' perps clear this floor on ' + (d.venues || [d.exchange]).join(' / ').toUpperCase() + '.';
    })
    .catch(function(){ btn.disabled = false; btn.textContent = 'How many coins pass?'; out.textContent = 'Failed.'; });
}

/* ── Scanner ──────────────────────────────────────────────────────────────── */
var cb = 0, cr = 0, cf = 0;
function bump(el, v){ el.textContent = v; }
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

var af = 'all', sortKey = 'score', pendR = [], hasScanned = false, seenSig = {};
function setTab(f){
  af = f;
  document.querySelectorAll('.ftab').forEach(function(b){ b.classList.toggle('active', b.dataset.f === f); });
}
function setF(btn){
  setTab(btn.dataset.f);
  showNo = af === 'all';             // picking All asks for the No rows too
  if(hasScanned) renderAll(pendR);   // before a scan, leave the welcome alone
}
document.querySelectorAll('th.sortable').forEach(function(th){
  th.addEventListener('click', function(){
    sortKey = th.dataset.sort;
    document.querySelectorAll('th.sortable').forEach(function(x){ x.classList.remove('sorted'); });
    th.classList.add('sorted');
    if(hasScanned) renderAll(pendR);
  });
});

var tot = 0, pRaf = null, pPend = null, RC = 201;
function setRing(pct, pair){
  document.getElementById('rfill').style.strokeDashoffset = RC * (1 - pct/100);
  document.getElementById('rpct').textContent = Math.round(pct) + '%';
  if(pair) document.getElementById('lpair').textContent = pair.replace('/USDT:USDT','').replace('/USDT','');
}
function setProgress(cur, t, pair){
  pPend = {cur:cur, t:t, pair:pair};
  if(pRaf) return;
  pRaf = requestAnimationFrame(function(){
    pRaf = null; var p = pPend; var pct = p.t > 0 ? (p.cur / p.t * 100) : 0;
    document.getElementById('pbFill').style.width = pct.toFixed(1) + '%';
    document.getElementById('pbLbl').textContent = p.cur + '/' + p.t + ' - ' + p.pair.replace('/USDT:USDT','').replace('/USDT','');
    if(p.t > 0) setRing(pct, p.pair);
  });
}
var sRaf = null, sPend = null;
function setStatus(lbl, live){
  sPend = {lbl:lbl, live:live};
  if(sRaf) return;
  sRaf = requestAnimationFrame(function(){
    sRaf = null; var s = sPend, el = document.getElementById('slbl');
    el.classList.add('fd');
    setTimeout(function(){ el.textContent = s.lbl; el.classList.remove('fd'); }, 120);
    document.getElementById('sdot').className = 'sdot' + (s.live ? ' live' : '');
  });
}

var LM = 4;
function pushLog(msg, cls){
  cls = cls || 'in';
  var c = document.getElementById('lc');
  var t = new Date().toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  c.querySelectorAll('.lcard').forEach(function(x){ x.classList.remove('in','dn','er'); x.classList.add('ol'); });
  var d = document.createElement('div');
  d.className = 'lcard ' + cls + ' ce';
  d.innerHTML = '<span class="lcard-m">' + msg + '</span><span class="lcard-t">' + t + '</span>';
  c.insertBefore(d, c.firstChild);
  requestAnimationFrame(function(){ requestAnimationFrame(function(){ d.classList.remove('ce'); }); });
  var all = c.querySelectorAll('.lcard');
  if(all.length > LM){
    var old = all[all.length - 1];
    old.classList.add('cx');
    setTimeout(function(){ if(old.parentNode) old.parentNode.removeChild(old); }, 420);
  }
}

function showLoader(){
  setRing(0, '');
  document.getElementById('lpair').textContent = 'Loading markets...';
  document.getElementById('sl').classList.add('vis');
  document.getElementById('sl').classList.remove('diss');
}
function dissolve(){
  var e = document.getElementById('sl');
  e.classList.add('diss');
  setTimeout(function(){ e.classList.remove('vis','diss'); }, 400);
}

var BM = 60, BOPS = ['pivot_detect','macd_align','swing_check','line_clean','atr_gate','candle_fetch','vol_ratio','diverge_scan'];
function rhex(n){ var s = '0x', h = '0123456789abcdef'; for(var i = 0; i < n; i++) s += h[Math.floor(Math.random()*16)]; return s; }
function pushBg(pair, isHit){
  var f = document.getElementById('sbgFeed'); if(!f) return;
  var t = new Date().toLocaleTimeString('en-GB', {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  var b = pair ? pair.replace('/USDT:USDT','').replace('/USDT','') : '';
  var op = BOPS[Math.floor(Math.random()*BOPS.length)];
  var txt = isHit ? ('[' + t + '] >>> HIT  ' + b.padEnd(10) + ' DIVERGENCE CONFIRMED  ' + rhex(4))
                  : ('[' + t + '] >   SCAN ' + b.padEnd(10) + ' ' + op + '  ' + rhex(4) + '  ' + rhex(4));
  if(f.children.length >= BM) f.removeChild(f.firstChild);
  var d = document.createElement('div');
  d.className = 'bgl' + (isHit ? ' hit' : '');
  d.textContent = txt;
  f.appendChild(d);
  requestAnimationFrame(function(){ requestAnimationFrame(function(){ d.classList.add('sh'); }); });
}
function resetBg(){ var f = document.getElementById('sbgFeed'); if(f) f.innerHTML = ''; }

function kv(k, v){ return '<dt>' + k + '</dt><dd>' + v + '</dd>'; }
/* One line per rule of the answer, for the primary timeframe: [class, text] with
   '' = passed, 'w' = waiting on it, 'x' = failed. The answer itself comes from the scan. */
function checks(r){
  var p = primary(r), tf = primaryTf(r), cfg = TFM[tf] || {}, a = [];
  var bars = p.bars_ago + ' completed ' + tfLabel(tf) + ' bar' + (p.bars_ago === 1 ? '' : 's') +
             ' since the signal (limit ' + cfg.fresh_h * 60 / cfg.minutes + ')';
  a.push(p.forming ? (p.trigger_first ? ['', 'Signal bar still forming, lower timeframe already triggered']
                                      : ['w', 'Signal bar still forming, no lower-timeframe trigger yet'])
                   : [p.fresh ? '' : 'x', bars]);
  a.push(p.grade === 'C' ? ['x', 'Grade C (' + p.score + ') is too weak'] : ['', 'Grade ' + p.grade + ' (' + p.score + ')']);
  a.push(p.regime === 'COUNTER TREND' && p.grade !== 'A' ? ['w', 'Counter trend needs grade A']
       : ['', p.regime.charAt(0) + p.regime.slice(1).toLowerCase()]);
  a.push(p.status === 'invalidated' ? ['x', 'Price traded through the stop']
       : p.status === 'ran' ? ['x', 'Already ran ' + p.r_reached + 'R']
       : p.status === 'extended' ? ['w', 'Price has moved ' + p.pct_chg + '% away from the signal']
       : ['', 'Price still near the signal']);
  var b = scanMeta.blocker;          /* undefined: a scan saved before the brief existed */
  if(b === null) a.push(['', 'No major US release on the calendar']);
  else if(b && scanTs){
    var h = (b.ts_ms - scanTs) / 3600000;
    a.push(h >= -1 && h <= cfg.macro_h
      ? ['w', calEsc(b.title) + (h > 0 ? ' was ' + calUntil(h * 3600000) + ' away at the scan' : ' had just printed at the scan')]
      : ['', 'No US release inside the ' + cfg.macro_h + 'h wait window at the scan']);
  }
  return a;
}
/* One block per diverging timeframe: chart, levels and setup, then the answer's
   checklist and the tags on the timeframe that supplies the answer. */
function tfBlock(r, tf){
  var s = r.signals[tf], t = s.targets || {}, main = tf === primaryTf(r), trig = s.targets_from === 'trigger';
  var src = '/chart?pair=' + encodeURIComponent(r.pair) + '&side=' + r.side + '&tf=' + tf;
  return '<div class="dgrid">' +
    '<div><a href="' + src + '" target="_blank" rel="noopener" title="Open the chart full size"><img class="chart" loading="lazy" alt="' +
      r.base + ' ' + tfLabel(tf) + ' chart needs a fresh scan" src="' + src + '"></a></div>' +
    '<div><span class="lbl">' + tfLabel(tf) + ' levels</span><dl class="kv num">' +
      kv('Now', fmtPx(r.current_price)) +
      kv('Entry', fmtPx(s.entry_ref_price) + (trig ? ' (trigger price)' : ' (signal price)')) +
      (s.trigger_price != null && !trig ? kv('Trigger', fmtPx(s.trigger_price)) : '') +
      kv('Stop', fmtPx(s.stop_price) + ' (' + s.risk_pct + '%)') +
      kv('1R', fmtPx(t['1r'])) + kv('2R', fmtPx(t['2r'])) + kv('3R', fmtPx(t['3r'])) +
      kv('Structure', fmtPx(t.structure)) + '</dl>' +
      '<span class="lbl">Setup</span><dl class="kv">' +
      kv('A', s.pivot1_time + ' @ ' + fmtPx(s.pivot1_price)) +
      kv('B', s.pivot2_time + ' @ ' + fmtPx(s.pivot2_price)) +
      kv('Signal', (s.forming ? 'bar still forming' : s.signal_time + ' (' + b2t(s.bars_ago, tf) + ')') + ' @ ' + fmtPx(s.signal_price)) +
      kv('Regime', '<span' + tip(s.regime) + '>' + s.regime + '</span> &middot; ' + (SC[s.status] || SC.active) +
        ' (' + (s.pct_chg >= 0 ? '+' : '') + s.pct_chg + '% in signal direction)') +
      kv('Data from', (r.exchange || '').toUpperCase() + ' &middot; ' + fmtVol(r.weekly_volume) + ' / week') +
      kv('Timeframes', r.tfs.map(tfLabel).join(' + ')) + '</dl></div>' +
    '<div>' + (main
      ? '<span class="lbl">Why this answer</span><ul class="chk">' +
        checks(r).map(function(c){ return '<li class="' + c[0] + '">' + c[1] + '</li>'; }).join('') + '</ul>' +
        '<p class="full">' + reason(r) + '</p>'
      : '<span class="lbl">' + tfLabel(tf) + ' also diverges</span><p class="full">Grade ' + s.grade + ' (' + s.score +
        '). The answer comes from ' + tfLabel(primaryTf(r)) + '.</p>') +
      '<span class="lbl">Score inputs</span><dl class="kv">' +
      kv('MACD A to B', s.pivot1_macd + ' &rarr; ' + s.pivot2_macd) +
      kv('Higher TF', s.htf_ok == null ? 'n/a' : s.htf_ok ? 'agrees' : 'no') +
      kv('OI into B', s.oi_chg_pct == null ? 'n/a' : s.oi_chg_pct + '%') +
      kv('BTC', s.btc_state || 'n/a') +
      kv('Volume', s.vol_dry ? 'drying' : 'not drying') +
      kv('Wick', s.wick ? 'yes' : 'no') + (s.triple ? kv('Triple', 'yes') : '') +
      (s.partial ? kv('Score', 'partial data') : '') + '</dl>' +
      (main ? '<span class="lbl">All tags</span><div>' + (tagChips(r).join('') || '<span class="calm">None</span>') +
              (r.context_error ? '<span class="calm" title="' + calEsc(r.context_error) + '"> Movers and OI context unavailable.</span>' : '') + '</div>' : '') +
    '</div></div>';
}
function detailRow(r){
  var tfs = [primaryTf(r)].concat(r.tfs.filter(function(t){ return t !== primaryTf(r); }));
  return '<tr class="det"><td colspan="7">' + tfs.map(function(tf){ return tfBlock(r, tf); }).join('') + '</td></tr>';
}
var openKey = null, showNo = false;
function buildRow(r){
  var p = primary(r), k = rowKey(r), tags = tagChips(r), more = tags.length - 2;
  return '<tr class="row" data-k="' + k + '" tabindex="0" aria-expanded="' + (openKey === k) + '">' +
    '<td class="coin"><b>' + r.base + '</b><span class="sub">' + tags.slice(0, 2).join('') + (more > 0 ? '<span class="tag">+' + more + '</span>' : '') + '</span></td>' +
    '<td>' + sideHtml(r) + '</td><td>' + gradeCell(p) + '</td><td class="why">' + evCell(r) + '</td>' +
    '<td class="num">' + fmtPx(p.entry_ref_price) + '</td>' +
    '<td class="num">' + fmtPx(p.stop_price) + '<span class="sub">risk ' + p.risk_pct + '%</span></td>' +
    '<td class="num">' + oiCell(r) + '</td></tr>';
}
function ageMin(r){ return primary(r).bars_ago * ((TFM[primaryTf(r)] && TFM[primaryTf(r)].minutes) || 240); }

/* No rows stay folded under the table until asked for: the fold button, or picking All. */
function renderAll(results){
  results = results || [];
  var rows = results.filter(function(d){ return matches(d, af); });
  rows.sort(function(a, b){
    if(sortKey === 'oi'){ var x = oiVal(a), y = oiVal(b); return (y == null ? -1e9 : y) - (x == null ? -1e9 : x); }
    return (EV_RANK[verdict(a)] - EV_RANK[verdict(b)]) ||
           (verdict(a) === 'skip' ? ageMin(a) - ageMin(b) : 0) ||   /* played-out rows: newest first */
           (b.score - a.score);
  });
  var h = '', no = 0;
  rows.forEach(function(r){
    if(verdict(r) === 'skip'){ no++; if(!showNo) return; }
    h += buildRow(r);
    if(openKey === rowKey(r)) h += detailRow(r);
  });
  if(!h) h = '<tr><td class="none" colspan="7">' + (!results.length ? 'No divergences found.'
    : af === 'all' ? 'No setup passes or is waiting right now.' : 'No setup matches this filter.') + '</td></tr>';
  document.getElementById('resB').innerHTML = h;
  document.getElementById('resWrap').style.display = '';
  var f = document.getElementById('fold');
  f.style.display = no ? '' : 'none';
  f.textContent = showNo ? 'Hide the ' + no + ' that did not pass' : 'Show ' + no + ' that did not pass, newest first';
  f.setAttribute('aria-expanded', showNo);
}
function rowEl(k){
  return Array.prototype.filter.call(document.querySelectorAll('#resB tr.row'), function(t){ return t.dataset.k === k; })[0];
}
function toggleRow(t){
  var k = t.dataset.k;
  openKey = openKey === k ? null : k;
  renderAll(pendR);
  if(rowEl(k)) rowEl(k).focus({preventScroll:true});
}
document.getElementById('resB').addEventListener('click', function(e){
  var t = e.target.closest('tr.row'); if(t) toggleRow(t);
});
document.getElementById('resB').addEventListener('keydown', function(e){
  if(e.target.matches('tr.row') && (e.key === 'Enter' || e.key === ' ')){ e.preventDefault(); toggleRow(e.target); }
});
document.getElementById('fold').addEventListener('click', function(){ showNo = !showNo; renderAll(pendR); });

/* ── The brief: what matters now, in sentences ────────────────────────────── */
var scanMeta = {};
/* Market and macro lines. Scans saved before the brief existed have no btc /
   blocker / macro in their meta: those lines are then left out, not guessed. */
function briefState(){
  var m = scanMeta, b = m.btc, now = Date.now(), s = '';
  if(b){
    var means = b.state === 'BULL' ? 'Longs are with the trend when the coin is also above its 200 EMA. Shorts are counter trend and need grade A.'
      : b.state === 'BEAR' ? 'Shorts are with the trend when the coin is also below its 200 EMA. Longs are counter trend and need grade A.'
      : 'BTC is neither trending up nor down, so reversals keep the normal bar: grade B or better.';
    s += '<p><b>Market is ' + (b.state === 'CHOP' ? 'in chop' : b.state.toLowerCase()) + '.</b> BTC 4H closed at <span class="num">' + fmtPx(b.close) +
      '</span>, ' + (b.close > b.ema200 ? 'above' : 'below') + ' its 200 EMA at <span class="num">' + fmtPx(b.ema200) +
      '</span>, with RSI <span class="num">' + Math.round(b.rsi) + '</span>. ' + means + '</p>';
  } else if(b === null) s += '<p><b>BTC data is unavailable.</b> Every setup is treated as counter trend.</p>';
  if(m.blocker){
    var h = (m.blocker.ts_ms - now) / 3600000, t = calEsc(m.blocker.title);
    s += h < -1 ? '<p><b>' + t + ' printed ' + calUntil(-h * 3600000) + ' ago.</b> Scan again to pick up the next release.</p>'
      : '<p><b>' + t + (h > 0 ? ' in ' + calUntil(h * 3600000) : ' printed ' + calUntil(-h * 3600000) + ' ago') + '.</b> ' +
        (h <= TFM['1h'].macro_h ? 'All entries wait until an hour after it prints.'
         : h <= TFM['4h'].macro_h ? '4H setups wait until it is out. 1H setups are clear for now.'
         : 'Far enough away that nothing waits on it yet.') + '</p>';
  } else if(m.blocker === null) s += '<p><b>No major US release on the calendar.</b> Nothing waits on macro.</p>';
  if(m.macro && m.macro.ts_ms > now) s += '<p>Also due: ' + calEsc(m.macro.title) + ' in ' + calUntil(m.macro.ts_ms - now) +
    '. Context only, it never blocks an entry.</p>';
  document.getElementById('briefState').innerHTML = s;
}
function renderBrief(){
  var R = pendR, byScore = function(a, b){ return b.score - a.score; };
  var yes = R.filter(function(r){ return verdict(r) === 'enter'; }).sort(byScore);
  var wait = R.filter(function(r){ return verdict(r) === 'wait'; }).sort(byScore);
  var hr = new Date().getHours(), hi = hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening';
  document.getElementById('briefH').textContent = hi + '. ' + (
    yes.length ? yes.length + (yes.length === 1 ? ' setup passes' : ' setups pass') + ' the checklist; the rest can wait.'
    : wait.length ? 'Nothing passes yet. ' + wait.length + (wait.length === 1 ? ' setup is' : ' setups are') + ' close.'
    : 'Nothing passes the checklist right now. No trade is a fine answer.');
  briefState();
  var list = yes.length ? yes.slice(0, 3) : wait.length ? wait.slice(0, 2)
           : R.slice().sort(function(a, b){ return ageMin(a) - ageMin(b); }).slice(0, 3);
  document.getElementById('briefLbl').textContent = yes.length ? 'Worth a look' : wait.length ? 'Closest to qualifying'
    : 'Nothing to act on. The three newest signals, and why each is a No';
  document.getElementById('picks').innerHTML = list.map(function(r){
    var p = primary(r);
    return '<button class="pick" data-k="' + rowKey(r) + '"><span class="head"><span class="coin">' + r.base + '</span><span>' + sideHtml(r) +
      '</span><span class="num gr">' + p.grade + ' ' + p.score + '</span><span class="old">' +
      (p.forming ? 'signal bar still forming' : 'signal ' + b2t(p.bars_ago, primaryTf(r))) + '</span></span>' +
      ansPill(r) + '<span class="why">' + reason(r) + '</span>' +
      '<span class="lv num">Entry <span>' + fmtPx(p.entry_ref_price) + '</span> &middot; Stop <span>' + fmtPx(p.stop_price) + '</span> (' + p.risk_pct +
      '%) &middot; 2R <span>' + fmtPx((p.targets || {})['2r']) + '</span> &middot; Now <span>' + fmtPx(r.current_price) + '</span></span></button>';
  }).join('');
  document.getElementById('briefPicks').style.display = list.length ? '' : 'none';
  document.getElementById('runBtn').textContent = 'Scan again';
}
/* A card opens its row in the table, unfolding or unfiltering whatever hides it. */
document.getElementById('picks').addEventListener('click', function(e){
  var b = e.target.closest('.pick'); if(!b) return;
  var r = pendR.filter(function(x){ return rowKey(x) === b.dataset.k; })[0]; if(!r) return;
  openKey = b.dataset.k;
  if(!matches(r, af)) setTab('all');
  if(verdict(r) === 'skip') showNo = true;
  renderAll(pendR);
  rowEl(openKey).scrollIntoView({block:'start'});
});

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
setInterval(function(){ updAge(); if(scanTs) briefState(); }, 60000);   /* the macro countdown runs live */

function loadLast(){
  fetch('/signals').then(function(r){ return r.json(); }).then(function(d){
    var rows = d.rows || [];
    if(!d.ts) return;                       /* never scanned: leave the welcome panel */
    pendR = rows; hasScanned = true;
    scanTs = new Date(d.ts).getTime();
    resetStats();
    rows.forEach(function(r){ if(verdict(r) !== 'skip') updStats(r.side, verdict(r) === 'enter'); });
    scanMeta = d.meta || {};
    document.getElementById('scan-audit').innerHTML = auditLine(d.meta);
    showResults(); updAge();
  }).catch(function(){});
}

function showResults(){ renderAll(pendR); renderBrief(); }
/* A scan that ended early: show what it found, or say that it stopped. The log above the table has the detail. */
function scanStopped(msg){
  document.getElementById('runBtn').disabled = false;
  dissolve();
  if(pendR.length){ scanTs = Date.now(); updAge(); setTimeout(showResults, 350); return; }
  document.getElementById('briefH').textContent = 'The scan stopped.';
  document.getElementById('briefState').innerHTML = '<p>' + calEsc(msg) + '</p>';
}

var es = null;
function startScan(){
  showPg('scan');
  if(es){ es.close(); es = null; }
  resetStats();
  document.getElementById('lc').querySelectorAll('.lcard').forEach(function(c){
    c.classList.add('cx');
    setTimeout(function(){ if(c.parentNode) c.parentNode.removeChild(c); }, 420);
  });
  document.getElementById('resWrap').style.display = 'none';
  document.getElementById('briefH').textContent = 'Scanning the market...';
  document.getElementById('briefState').innerHTML = '';
  document.getElementById('briefPicks').style.display = 'none';
  document.getElementById('pgSec').style.display = '';
  document.getElementById('runBtn').disabled = true;
  tot = 0; setProgress(0, 0, ''); setStatus('Scanning...', true);
  showLoader(); resetBg(); pendR = []; seenSig = {}; hasScanned = true;
  scanTs = null; updAge(); scanMeta = {}; openKey = null;
  document.getElementById('scan-audit').innerHTML = '';

  var exch = document.getElementById('exchange').value;
  var str  = document.querySelector('#strG .tbtn.active').dataset.val;
  var spd  = document.querySelector('#spdG .tbtn.active').dataset.val;
  var vol  = (parseInt(document.getElementById('minVol').value) || 70) * 1e6;
  var mp   = parseInt(document.getElementById('maxPairs').value) || 400;
  var syms = document.getElementById('symOverride').value.trim();

  var url = '/scan?exchange=' + exch + '&strictness=' + str +
            '&speed=' + spd + '&min_weekly_vol=' + vol + '&max_pairs=' + mp;
  if(syms) url += '&symbols=' + encodeURIComponent(syms);

  es = new EventSource(url);
  es.addEventListener('log', function(e){ pushLog(JSON.parse(e.data).msg, 'in'); });
  es.addEventListener('total', function(e){ tot = JSON.parse(e.data).total; });
  es.addEventListener('progress', function(e){
    var d = JSON.parse(e.data);
    setProgress(d.current, tot, d.pair);
    setStatus(d.current + '/' + tot, true);
    pushBg(d.pair, false);
  });
  es.addEventListener('result', function(e){
    var d = JSON.parse(e.data);
    var k = d.pair + '|' + d.side;
    if(seenSig[k]) return;            // duplicate guard, see es.onerror
    seenSig[k] = 1;
    if(verdict(d) !== 'skip') updStats(d.side, verdict(d) === 'enter');
    pendR.push(d);
    pushBg(d.pair, true);
  });
  es.addEventListener('done', function(e){
    var d = JSON.parse(e.data);
    es.close();
    document.getElementById('runBtn').disabled = false;
    setProgress(tot, tot, '');
    setStatus('Done - ' + d.hits + ' row' + (d.hits !== 1 ? 's' : ''), false);
    var m = d.meta || {};
    pushLog('Scan complete - ' + d.hits + ' divergence row(s) from ' + (m.scanned || 0) + ' pairs analysed', 'dn');
    scanTs = Date.now(); updAge(); scanMeta = m;
    document.getElementById('scan-audit').innerHTML = auditLine(m);
    if(m.vol_rejected) pushLog(m.vol_rejected + ' pair(s) fell short on 7-day volume', 'in');
    if(m.no_data || m.errors) pushLog(((m.no_data||0) + (m.errors||0)) + ' pair(s) could not be analysed (no candles / error)', 'er');
    setRing(100, '');
    setTimeout(function(){
      dissolve();
      /* the progress and log give way to the brief; the audit line under the table keeps the counts */
      setTimeout(function(){ document.getElementById('pgSec').style.display = 'none'; showResults(); }, 350);
    }, 500);
  });
  es.addEventListener('error', function(e){
    if(!e.data) return;  // native connection errors handled by es.onerror
    var m = 'Scan error';
    try { m = JSON.parse(e.data).msg; } catch(x){}
    pushLog(m, 'er'); es.close();
    setStatus('Error', false);
    scanStopped(m);
  });
  es.onerror = function(){
    if(es.readyState === EventSource.CLOSED) return;
    // EventSource auto-reconnects on any transport drop, and a reconnect here
    // re-issues GET /scan - a second full market scan, sharing the exchange
    // quota with the first and appending duplicate rows to pendR. Close it and
    // let the user decide to re-run.
    es.close();
    pushLog('Connection lost - scan stopped. Run the scan again to restart.', 'er');
    setStatus('Disconnected', false);
    scanStopped('Connection lost. Run the scan again to restart.');
  };
}

/* Before any scan the brief is the welcome: headline, one sentence, and the Run Scan button under it. */
function renderWelcome(){
  var h = new Date().getHours();
  var subs = ['Ready to find divergences.',
              '4H and 1H, regular divergences only.',
              'Every liquid perp, one scan, one table.',
              'Set the volume floor and hit Run.'];
  document.getElementById('briefH').textContent = (h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening') + ', Kevin';
  document.getElementById('briefState').innerHTML = '<p>' + subs[Math.floor(Math.random()*subs.length)] + '</p>';
}

/* ── Calendar ─────────────────────────────────────────────────────────────── */
var _cal = {events: [], meta: {}, ts: null}, _cf = 'movers', _calSel = null;
var CAL_DOW = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
var CAL_MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
/* Short names for the grid chips; the full title lives in the tooltip and the day table. */
var CAL_SHORT = [
  [/^Non-Farm Employment Change$/i, 'NFP'],
  [/^Average Hourly Earnings.*$/i, 'Avg Earnings'],
  [/^Unemployment Rate$/i, 'Unemployment'],
  [/^Federal Funds Rate$/i, 'Fed Rate'],
  [/^FOMC Press Conference$/i, 'FOMC Presser'],
  [/^FOMC Meeting Minutes$/i, 'FOMC Minutes'],
  [/^FOMC Economic Projections$/i, 'FOMC Projections'],
  [/^Fed Chair (\w+) (Speaks|Testifies)$/i, '$1 $2'],
  [/^Core PCE Price Index.*$/i, 'Core PCE'],
  [/^ISM Manufacturing PMI$/i, 'ISM Mfg'],
  [/^ISM Services PMI$/i, 'ISM Services'],
  [/^Main Refinancing Rate$/i, 'ECB Rate'],
  [/^Official Bank Rate$/i, 'BoE Rate'],
  [/^BOJ Policy Rate$/i, 'BoJ Rate'],
  [/^(Advance|Prelim|Final) GDP.*$/i, 'GDP ($1)'],
  [/^Core Retail Sales.*$/i, 'Core Retail'],
  [/^Retail Sales.*$/i, 'Retail Sales']
];
function calEsc(s){ return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function calShort(t){
  for(var i = 0; i < CAL_SHORT.length; i++){ if(CAL_SHORT[i][0].test(t)) return t.replace(CAL_SHORT[i][0], CAL_SHORT[i][1]); }
  return t.replace(/\s+(m\/m|q\/q|y\/y)$/i, '');
}
function calPad(n){ return (n < 10 ? '0' : '') + n; }
function calHM(d){ return calPad(d.getHours()) + ':' + calPad(d.getMinutes()); }
function calUTC(d){ return calPad(d.getUTCHours()) + ':' + calPad(d.getUTCMinutes()) + ' UTC'; }
function calDayKey(d){ return d.getFullYear() + '-' + calPad(d.getMonth() + 1) + '-' + calPad(d.getDate()); }
function calDayLabel(d){ return CAL_DOW[d.getDay()] + ' ' + d.getDate() + ' ' + CAL_MON[d.getMonth()]; }
/* ForexFactory's own permalink shape, e.g. calendar?day=sep23.2026. If the
   param is ever rejected the site just opens on the current calendar, so a
   miss costs a click rather than a broken link. */
function calFFUrl(d){
  return 'https://www.forexfactory.com/calendar?day=' +
         CAL_MON[d.getMonth()].toLowerCase() + d.getDate() + '.' + d.getFullYear();
}
function calUntil(ms){
  if(ms <= 0) return 'now';
  var m = Math.round(ms / 60000), d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60), mm = m % 60;
  if(d > 0) return d + 'd ' + h + 'h';
  if(h > 0) return h + 'h ' + mm + 'm';
  return mm + 'm';
}
function calFiltered(){
  return _cal.events.filter(function(e){ return _cf === 'movers' ? e.mover : e.impact === 'High'; });
}
/* The banner always uses the curated list, whatever the grid filter is. */
function calNextMover(now){
  var n = null;
  _cal.events.forEach(function(e){
    if(!e.mover) return;
    var t = new Date(e.ts).getTime();
    if(t > now && (n === null || t < n.t)) n = {e: e, t: t};
  });
  return n;
}
function calBanner(now){
  var n = calNextMover(now);
  var el = document.getElementById('cal-next'), strip = document.getElementById('dash-cal');
  if(!n){
    var msg = _cal.events.length ? 'No market movers on the schedule through ' + (_cal.meta.covered_to ? calDayLabel(new Date(_cal.meta.covered_to + 'T12:00:00Z')) : 'the current horizon') + '.' : 'Calendar not loaded yet.';
    el.className = 'cal-next';
    el.innerHTML = '<span class="cal-next-lbl">Next market mover</span><span class="cal-next-meta">' + msg + '</span>';
    if(strip) strip.style.display = 'none';
    return;
  }
  var sibs = _cal.events.filter(function(e){ return e.mover && e.ts === n.e.ts; });
  var d = new Date(n.t), soon = (n.t - now) <= 86400000;
  var names = sibs.map(function(e){ return calShort(e.title); }).join(' + ');
  var when = calDayLabel(d) + ' ' + calHM(d);
  var vals = sibs.map(function(e){
    return '<b>' + calEsc(calShort(e.title)) + '</b> fc ' + (calEsc(e.forecast) || '&mdash;') + ' / prev ' + (calEsc(e.previous) || '&mdash;');
  }).join(' &nbsp;&middot;&nbsp; ');
  el.className = 'cal-next' + (soon ? ' soon' : '');
  el.innerHTML = '<span class="cal-next-lbl">Next market mover</span>' +
    '<span class="cal-next-ttl">' + calEsc(names) + ' <span style="color:var(--tx3);font-weight:400">' + calEsc(n.e.currency) + '</span></span>' +
    '<span class="cal-next-in">in ' + calUntil(n.t - now) + '</span>' +
    '<span class="cal-next-meta" title="' + calUTC(d) + '">' + when + ' &nbsp;&middot;&nbsp; ' + vals + '</span>';
  if(strip){
    strip.className = 'cal-strip' + (soon ? ' soon' : '');
    strip.style.display = 'flex';
    strip.innerHTML = '<span class="lbl">Next market mover</span><b>' + calEsc(names) + '</b>' +
      '<span title="' + calUTC(d) + '">' + calEsc(n.e.currency) + ' &middot; ' + when + '</span>' +
      '<span class="in">in ' + calUntil(n.t - now) + '</span>' +
      '<a onclick="showPg(\'calendar\')">Calendar &rarr;</a>';
  }
}
function calGrid(now){
  var g = document.getElementById('cal-grid'); g.innerHTML = '';
  var byDay = {};
  calFiltered().forEach(function(e){ var k = calDayKey(new Date(e.ts)); (byDay[k] = byDay[k] || []).push(e); });
  var start = new Date(now); start.setHours(0, 0, 0, 0);
  if(_calSel === null) _calSel = calDayKey(start);
  /* Past the horizon the feeds actually cover, a blank cell means "no data", not "quiet".
     This is the requested range, not the last event: an empty Saturday inside a published week is genuinely quiet. */
  var coveredTo = _cal.meta.covered_to || '';
  for(var i = 0; i < 14; i++){
    var d = new Date(start.getTime()); d.setDate(start.getDate() + i);
    var k = calDayKey(d), list = byDay[k] || [];
    var soon = list.some(function(e){ var t = new Date(e.ts).getTime(); return e.mover && t > now && t - now <= 86400000; });
    var cell = document.createElement('div');
    cell.className = 'cal-day' + (i === 0 ? ' today' : '') + (soon ? ' soon' : '') + (k === _calSel ? ' sel' : '');
    cell.dataset.k = k;
    var html = '<div class="cal-dh"><span class="cal-dow">' + CAL_DOW[d.getDay()] + '</span>' +
               '<span class="cal-dn">' + d.getDate() + ((d.getDate() === 1 || i === 0) ? ' ' + CAL_MON[d.getMonth()] : '') + '</span></div>';
    if(list.length === 0){
      var nodata = coveredTo && k > coveredTo;
      html += '<div class="cal-quiet">' + (nodata ? 'no data yet' : 'quiet') + '</div>';
    } else {
      var groups = [], last = null;
      list.forEach(function(e){
        if(last && last.ts === e.ts){ last.items.push(e); } else { last = {ts: e.ts, items: [e]}; groups.push(last); }
      });
      groups.forEach(function(gr){
        var t = new Date(gr.ts), done = t.getTime() < now, mv = gr.items.some(function(e){ return e.mover; });
        var names = gr.items.slice(0, 3).map(function(e){ return calEsc(calShort(e.title)); }).join(' + ') +
                    (gr.items.length > 3 ? ' +' + (gr.items.length - 3) : '');
        var curs = []; gr.items.forEach(function(e){ if(curs.indexOf(e.currency) < 0) curs.push(e.currency); });
        html += '<div class="cal-ev' + (mv ? ' mv' : '') + (done ? ' done' : '') + '" title="' +
                calEsc(gr.items.map(function(e){ return e.title; }).join(', ')) + ' (' + calUTC(t) + ')">' +
                '<span class="cal-ev-t">' + calHM(t) + '</span>' + names + '<span class="cal-ev-c">' + calEsc(curs.join('/')) + '</span></div>';
      });
    }
    cell.innerHTML = html;
    cell.addEventListener('click', function(){ _calSel = this.dataset.k; renderCalendar(); });
    g.appendChild(cell);
  }
}
function calDetail(now){
  var sel = _calSel, rows = calFiltered().filter(function(e){ return calDayKey(new Date(e.ts)) === sel; });
  var p = sel.split('-'), d = new Date(+p[0], +p[1] - 1, +p[2]);
  var ffl = document.getElementById('cal-ff-day');
  ffl.href = calFFUrl(d);
  ffl.textContent = 'Open ' + calDayLabel(d) + ' on ForexFactory \u2197';
  document.getElementById('cal-day-ttl').textContent =
    calDayLabel(d) + ' \u2014 ' + (rows.length ? rows.length + ' scheduled' : 'nothing scheduled') +
    (_cf === 'movers' ? ' (market movers)' : ' (all high-impact)');
  var tb = document.getElementById('calB'); tb.innerHTML = '';
  document.getElementById('calT').style.display = rows.length ? 'table' : 'none';
  document.getElementById('calEmpty').style.display = rows.length ? 'none' : '';
  rows.forEach(function(e){
    var t = new Date(e.ts), done = t.getTime() < now;
    var tr = document.createElement('tr'); if(done) tr.style.opacity = '.5';
    var impCls = e.impact === 'High' ? ' hi' : '';
    tr.innerHTML =
      '<td class="num"><span title="' + calUTC(t) + '">' + calHM(t) + '</span></td>' +
      '<td class="lo">' + calEsc(e.currency) + '</td>' +
      '<td><b>' + calEsc(e.title) + '</b>' +
        (e.mover ? ' &nbsp;<span class="cal-imp hi" title="On the curated market-mover list">MOVER</span>' : '') + '</td>' +
      '<td><span class="cal-imp' + impCls + '">' + calEsc(e.impact) + '</span></td>' +
      '<td class="num">' + (calEsc(e.forecast) || '&mdash;') + '</td>' +
      '<td class="num">' + (calEsc(e.previous) || '&mdash;') + '</td>' +
      '<td class="num">' + (calEsc(e.actual) || '&mdash;') + '</td>';
    tb.appendChild(tr);
  });
}
function calMetaLine(){
  var m = _cal.meta || {}, bits = [];
  var SRC = {'jblanked+forexfactory': 'ForexFactory + JBlanked', 'jblanked': 'JBlanked only', 'forexfactory': 'ForexFactory feed'};
  bits.push(SRC[m.source] || 'no source');
  if(_cal.ts) bits.push('updated ' + Math.round((Date.now() - new Date(_cal.ts)) / 60000) + ' min ago');
  if(m.covered_to) bits.push('through ' + calDayLabel(new Date(m.covered_to + 'T12:00:00Z')));
  document.getElementById('cal-meta').textContent = bits.join(' \u00b7 ');
  var warn = [], info = [];
  if(m.fallback) warn.push(m.fallback + ': only the current week is shown. A JBlanked key with credits in .env extends the horizon.');
  if(/credit/i.test(m.error || '')) warn.push('JBlanked needs credits on the account before it can extend the calendar past this week. ForexFactory covers the current week either way.');
  if(m.error) warn.push('Last fetch problem: ' + m.error);
  if(m.time_check_h) info.push('JBlanked clock was ' + (m.time_check_h > 0 ? '+' : '') + m.time_check_h + 'h vs ForexFactory; its events were corrected.');
  var a = document.getElementById('cal-audit');
  a.className = 'audit' + (warn.length ? ' warn' : '');
  a.innerHTML = (warn.length ? '&#9888; ' : '') + warn.concat(info).map(calEsc).join('<br>');
}
function renderCalendar(){
  var now = Date.now();
  calBanner(now); calGrid(now); calDetail(now); calMetaLine();
}
function calApply(d){ _cal = {events: d.events || [], meta: d.meta || {}, ts: d.ts}; renderCalendar(); }
function loadCalendar(){
  fetch('/calendar').then(function(r){ return r.json(); }).then(calApply)
    .catch(function(){ document.getElementById('cal-meta').textContent = 'Failed to load calendar.'; });
}
function refreshCalendar(){
  var b = document.getElementById('calRefreshBtn'); b.disabled = true;
  fetch('/calendar?refresh=1').then(function(r){ return r.json(); }).then(calApply)
    .catch(function(){}).then(function(){ b.disabled = false; });
}
function setCF(btn){
  document.querySelectorAll('.sig-ftab[data-cf]').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active'); _cf = btn.dataset.cf; renderCalendar();
}
setInterval(function(){ if(_cal.events.length) calBanner(Date.now()); }, 60000);

document.addEventListener('DOMContentLoaded', function(){
  renderWelcome();
  loadLast();
  loadCalendar();
  var pg = (location.hash || '').slice(1);
  if(pg && document.getElementById('pg-' + pg)) showPg(pg);   /* deep link, e.g. /#calendar */
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="MACD wave divergence screener")
    ap.add_argument("--backtest", action="store_true",
                    help="replay the stored history and print win rates instead of starting the dashboard")
    ap.add_argument("--exchange", default="binance", choices=sorted(ALLOWED_EXCHANGES))
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
