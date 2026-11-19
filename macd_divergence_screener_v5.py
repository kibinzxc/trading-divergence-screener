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
optional early entry from the lower timeframe (ltf_trigger). `--backtest`
replays the same detector bar by bar (backtest_pair) and reports win rate and
R per grade and regime. Everything below about pivots, PIVOT_SYNC, the clean
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

ALLOWED_EXCHANGES = {"okx", "binance", "bybit"}
DEFAULT_TYPE      = {"okx": "swap", "binance": "future", "bybit": "swap"}
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
LTF_BARS_15M  = 300   # 15m history, for the 1H signal's early-entry trigger

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


def finish_signal(sig, htf_ok=None, oi_chg=None, fund=None, btc_state=None, trigger=None):
    """Score, grade, regime, trigger, tags and targets.

    Called once without context so a signal is always complete, and again by
    add_context() once the scan has fetched the outside data. `trigger` is
    (price, close time in ms of the trigger bar) or None.
    """
    sig["score"], sig["grade"], sig["partial"] = score_signal(sig, htf_ok, oi_chg, fund, btc_state)
    sig["htf_ok"] = htf_ok
    sig["oi_chg_pct"] = None if oi_chg is None else round(oi_chg, 1)
    sig["btc_state"] = btc_state
    sig["regime"] = regime_tag(sig["side"], sig["above_ema200"], btc_state)
    sig["trigger_price"] = px(trigger[0]) if trigger else None
    # "first" = the trigger bar closed before the signal bar did
    sig["trigger_first"] = bool(trigger and (sig["forming"] or trigger[1] < sig["signal_close_ms"]))
    sig["tags"] = [t for t, on in (("TRIPLE", sig["triple"]),
                                   ("W-PATTERN", sig["bump_frac"] >= 0.5),
                                   ("LTF TRIGGER", bool(trigger)),
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


def add_context(sig, htf, ltf, oi_hist, fund, btc):
    """Re-grade a signal with everything fetched for it. Any argument may be
    None; that component is then unavailable and leaves the score's denominator."""
    cfg = TF_CFG[sig["tf"]]
    close_ts = pd.Timestamp(sig["signal_close_ms"], unit="ms")
    trig = None
    if ltf is not None and sig["fresh"]:
        trig = ltf_trigger(ltf, sig["side"], pd.Timestamp(sig["b_ms"], unit="ms"),
                           pd.Timestamp(sig["signal_ms"], unit="ms"), cfg["ltf_minutes"])
    return finish_signal(sig, htf_agrees(htf, close_ts, cfg["htf_minutes"], sig["side"]),
                         oi_change_pct(oi_hist, sig["b_start_ms"], sig["b_ms"]),
                         fund, btc_state_at(btc, close_ts), trig)


def grade_signals(sigs, df4, df1, d1, df15, oi_d, oi_h, fund, btc):
    """Give each signal the context that belongs to its timeframe.

    4H signal: higher timeframe = 1D (d1), lower = 1H (df1).
    1H signal: higher timeframe = 4H (df4), lower = 15m (df15).
    Open interest: hourly history when it covers wave B, else daily.
    Any frame may be None.
    """
    for s in sigs:
        four = s["tf"] == "4h"
        hist = oi_h if oi_change_pct(oi_h, s["b_start_ms"], s["b_ms"]) is not None else oi_d
        add_context(s, d1 if four else df4, df1 if four else df15, hist, fund, btc)
    return sigs


# ── Entry verdict ──────────────────────────────────────────────────────────────
EVENT_WAIT_H = 24     # a curated macro print inside this many hours means wait


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


def fetch_btc(ex, limiter):
    """BTC 4H RSI 14, close and 200 EMA on completed bars, for the BTC state.
    None when unavailable."""
    df = drop_forming_candle(
        fetch_df_deep(ex, limiter, "BTC/USDT:USDT", "4h", TF_CFG["4h"]["bars"]), 240)
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


def build_rows(sigs, mover=None, flow=None, context_error=None,
               hours_to_event=None, event_title=""):
    """One row per side for ONE pair's signals across timeframes.

    The primary signal — the one whose status, levels and entry verdict the
    table shows — is the one with the best entry verdict (enter, then wait,
    then skip); on a tie the earlier timeframe in TF_CFG order (4H) wins.
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
    close. Not replayed: open interest (no bar-by-bar history), so that
    component leaves the score's denominator, and the lower-timeframe trigger.
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
                # Graded as of the signal bar's close, exactly like add_context.
                close_ts = pd.Timestamp(sig["signal_close_ms"], unit="ms")
                state = btc_state_at(btc, close_ts)
                score, grade, _ = score_signal(sig, htf_agrees(htf, close_ts, cfg["htf_minutes"], side),
                                               btc_state=state)
                trades.append(dict(
                    trade_outcome(df, i, side, entry, stop), tf=tf, side=side,
                    score=score, grade=grade, time=str(view.index[-1]),
                    regime=regime_tag(side, sig["above_ema200"], state)))
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
        if not pairs:
            print(f"No matching USDT perps on {exid.upper()}.")
            return
    else:
        tickers = ex.fetch_tickers()
        pairs = sorted(perps, key=lambda s: usd_volume(tickers.get(s)) or 0, reverse=True)[:n_pairs]

    def load(sym, tf, bars, minutes):
        df = drop_forming_candle(fetch_df_deep(ex, limiter, sym, tf, bars), minutes)
        return with_indicators(df) if df is not None and len(df) >= 80 else None

    btc = fetch_btc(ex, limiter)
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
          "lower-timeframe trigger are not replayed;\nthe OI component is left out of the "
          "score and the rest rescaled, as the live scan does when OI is missing.")


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
        "__TF_CFG_JSON__": json.dumps({k: {"label": v["label"], "minutes": v["minutes"]}
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
    exid = request.args.get("exchange", "okx")
    floor = float(request.args.get("min_weekly_vol", DEFAULT_MIN_WEEKLY_VOL))
    try:
        ex = make_exchange(exid)
        mkts = ex.load_markets()
        tickers = ex.fetch_tickers()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    rows = []
    for sym in usdt_perps(mkts):
        base = sym.split("/")[0]
        v = usd_volume(tickers.get(sym))
        if v:
            rows.append({"base": base, "sym": sym, "wk": v * 7.0})

    rows.sort(key=lambda r: r["wk"], reverse=True)
    passed = [r for r in rows if r["wk"] >= floor]
    return jsonify({
        "exchange": exid,
        "total_perps": len(rows),
        "passing": len(passed),
        "floor": floor,
        "top": passed[:60],
        "note": "Weekly figure here is 24h x 7 (a fast estimate). The scan "
                "measures true rolling 7-day volume from the candles.",
    })


@app.route("/scan")
def scan_route():
    exid       = request.args.get("exchange", "okx")
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

        perps = usdt_perps(mkts)
        yield sse("log", {"msg": f"{len(perps)} active linear USDT perps on {exid.upper()}"})

        yield sse("log", {"msg": "Fetching 24h volumes (one bulk call)..."})
        try:
            tickers = main_ex.fetch_tickers()
        except Exception as e:
            yield sse("error", {"msg": f"volume fetch failed: {e}"}); return

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

        # Cheap pre-gate on the ticker's 24h volume. It exists only to avoid
        # fetching 700 bars for obviously dead coins — the REAL gate is the
        # 7-day measurement from the candles further down. PREGATE_SLACK is
        # deliberately loose (a third of the daily-average equivalent) because
        # a coin can trade most of its week in one day: a listing that did
        # $190M five days ago and $2M since has a genuinely large 7-day volume
        # and would be thrown away by a tight 24h proxy.
        vol24 = {}
        for s in perps:
            v = usd_volume(tickers.get(s))
            if v:
                vol24[s] = v
        no_ticker = len(perps) - len(vol24)
        if no_ticker:
            yield sse("log", {"msg": f"{no_ticker} perp(s) report no usable 24h volume — not scanned"})

        pre_floor = (min_vol / 7.0) * PREGATE_SLACK
        pairs = [s for s in vol24 if vol24[s] >= pre_floor]
        pairs.sort(key=lambda s: vol24[s], reverse=True)

        if only_syms:
            wanted = {s.split("/")[0].split(":")[0] for s in only_syms}
            pairs = [s for s in perps if s.split("/")[0] in wanted]
            found = {s.split("/")[0] for s in pairs}
            missing = sorted(wanted - found)
            msg = f"Symbol override: scanning {len(pairs)} pair(s)"
            if missing:
                # Silently ignoring a typo'd or delisted ticker is the worst
                # outcome here — the user reads an empty table as "no setup".
                msg += f" · NOT LISTED on {exid.upper()}: {', '.join(missing)}"
            yield sse("log", {"msg": msg})
            if not pairs:
                yield sse("error", {"msg": f"None of those symbols are listed as USDT perps "
                                           f"on {exid.upper()}: {', '.join(sorted(wanted))}"})
                return
        else:
            dropped = len(vol24) - len(pairs)
            if len(pairs) > max_pairs:
                yield sse("log", {"msg": f"{len(pairs)} pairs cleared the pre-filter — "
                                         f"capping at the {max_pairs} most liquid by 24h volume"})
                pairs = pairs[:max_pairs]
            yield sse("log", {"msg": f"{len(pairs)} candidates · {dropped} too illiquid to be worth "
                                     f"fetching · 7-day volume checked per pair next"})

        spd = SPEED[speed]
        limiter = RateLimiter(spd["rps"].get(exid, 8))
        btc = fetch_btc(main_ex, limiter)
        if btc is None:
            yield sse("log", {"msg": "BTC 4H candles unavailable: every setup is tagged COUNTER TREND "
                                     "and scored without the BTC component (PARTIAL DATA)"})
        _charts.clear()
        workers = spd["workers"]
        yield sse("log", {"msg": f"Scanning {len(pairs)} pairs on 4H + 1H · "
                                 f"{strictness} · {workers} workers"})
        yield sse("total", {"total": len(pairs)})

        tls = threading.local()

        def worker_ex():
            ex = getattr(tls, "ex", None)
            if ex is None:
                ex = make_exchange(exid, rate_limit=False)
                ex.set_markets(mkts)          # reuse markets — no extra API call
                tls.ex = ex
            return ex

        q = queue.Queue()
        counter = {"n": 0, "analysed": 0, "vol_reject": 0, "no_data": 0, "errors": 0,
                   "no_1h": 0}
        clock = threading.Lock()

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

                f4 = forming_df(raw, df4)
                df4 = with_indicators(df4)
                if len(df4) < 80:
                    with clock:
                        counter["no_data"] += 1
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

                with clock:
                    counter["analysed"] += 1

                sigs = (signals_for_tf(df4, "4h", rules, live_price, sym, wk_vol, f4) +
                        signals_for_tf(df1, "1h", rules, live_price, sym, wk_vol, f1))
                if not sigs:
                    return
                # Everything below is fetched only for pairs worth a row.
                mover, flow, cerr, d1, oi_d = pair_context(ex, limiter, sym, live_price,
                                                           funding.get(sym), wk_vol)
                d1 = with_indicators(d1) if d1 is not None else None
                oi_h = df15 = None
                if any(s["fresh"] for s in sigs):       # stale signals answer No whatever the grade
                    oi_h = fetch_oi_history(ex, limiter, sym, 100, tf="1h")
                if any(s["tf"] == "1h" and s["fresh"] for s in sigs):
                    df15 = drop_forming_candle(
                        fetch_df_deep(ex, limiter, sym, "15m", LTF_BARS_15M), 15)
                    df15 = with_indicators(df15) if df15 is not None and len(df15) >= 80 else None
                grade_signals(sigs, df4, df1, d1, df15, oi_d, oi_h, funding.get(sym), btc)
                for s in sigs:
                    _charts[f"{sym}|{s['side']}|{s['tf']}"] = (s.pop("_df"), s.pop("_d"), dict(s))
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
                    "complete": done.is_set() and q.empty()}
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
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#090D18;--sf:#0F1623;--sf2:#141D2E;--sf3:#1A2438;
  --bd:#1E2D42;--bd2:#253449;
  --tx:#CDD6E8;--tx2:#8A9BB5;--tx3:#4F6380;
  --bull:#34D399;--bull-bg:#0D2E1F;
  --bear:#FB7185;--bear-bg:#2D0F18;
  --acc:#F0608A;--fresh:#FBBF24;--fresh-bg:#2D1F06;
  --mono:'JetBrains Mono',monospace;--sans:'Inter',system-ui,sans-serif;
  --ease:cubic-bezier(.22,.68,0,1.2);--eout:cubic-bezier(0,.55,.45,1);
}
html,body{height:100%;background:var(--bg);color:var(--tx);font-family:var(--sans);font-size:14px;line-height:1.5}
.shell{display:grid;grid-template-rows:48px 1fr;grid-template-columns:minmax(0,1fr);height:100vh;overflow:hidden}
.topbar{display:flex;align-items:center;background:var(--sf);border-bottom:1px solid var(--bd);z-index:20}
.t-logo{font-family:var(--mono);font-size:12px;font-weight:600;letter-spacing:.12em;color:var(--acc);padding:0 18px;flex-shrink:0}
.t-div{width:1px;height:16px;background:var(--bd2)}
.nav{display:flex;height:48px;padding:0 8px;min-width:0;overflow-x:auto;scrollbar-width:none}
.nav-btn{height:100%;padding:0 18px;font-size:12px;font-weight:500;color:var(--tx3);cursor:pointer;border:none;border-bottom:2px solid transparent;background:none;transition:color .15s,border-color .15s;letter-spacing:.04em;flex-shrink:0;white-space:nowrap}
.nav-btn:hover{color:var(--tx2)}
.nav-btn.active{color:var(--tx);border-bottom-color:var(--acc)}
.t-right{margin-left:auto;display:flex;align-items:center;gap:8px;padding-right:18px;flex-shrink:0}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--tx3);transition:background .4s,box-shadow .4s}
.sdot.live{background:var(--bull);box-shadow:0 0 8px var(--acc);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
.slbl{font-family:var(--mono);font-size:11px;color:var(--tx3);transition:opacity .2s}
.slbl.fd{opacity:.3}
.page{display:none}
.page.active{display:flex;flex-direction:column}
#pg-scan.active{display:grid;grid-template-columns:286px 1fr;overflow:hidden}
.wq{display:inline-flex;align-items:center;padding:3px 7px;border-radius:4px;font-family:var(--mono);font-size:10px;font-weight:700;white-space:nowrap}
.wq-sc{background:var(--sf3);color:var(--tx2);border:1px solid var(--bd2)}
.wq-cap{background:var(--sf3);color:var(--tx3);border:1px solid var(--bd2)}
.wflag{display:inline-block;padding:2px 6px;border-radius:3px;font-family:var(--mono);font-size:9px;font-weight:600;margin-right:4px;white-space:nowrap}
.wf-spot{background:var(--bull-bg);color:var(--bull);border:1px solid #1a4020}
.wf-crowd{background:#2D1500;color:#FB923C;border:1px solid #4a2800}
.wf-squeeze{background:var(--fresh-bg);color:var(--fresh);border:1px solid #4a3500}
.wf-hioi{background:var(--sf3);color:var(--tx2);border:1px solid var(--bd2)}
.wbar{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.wbar select{width:130px}
.wbar .rbtn{width:auto;padding:8px 22px}
.wprog{font-family:var(--mono);font-size:11px;color:var(--tx3)}
.sb{background:var(--sf);border-right:1px solid var(--bd);display:flex;flex-direction:column;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--bd2) transparent}
.ss{padding:13px 15px;border-bottom:1px solid var(--bd)}
.fl{font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--tx3);margin-bottom:7px}
.hint{font-family:var(--mono);font-size:9px;color:var(--tx3);line-height:1.6;margin-top:7px}
select,input[type=text]{width:100%;background:var(--sf2);border:1px solid var(--bd2);color:var(--tx);font-family:var(--mono);font-size:12px;padding:7px 10px;border-radius:5px;outline:none;appearance:none;transition:border-color .2s}
select:focus,input[type=text]:focus{border-color:var(--acc)}
input[type=number]{-moz-appearance:textfield}
input[type=number]::-webkit-inner-spin-button,input[type=number]::-webkit-outer-spin-button{-webkit-appearance:none}
.nw{display:flex;align-items:center;background:var(--sf2);border:1px solid var(--bd2);border-radius:5px;overflow:hidden;transition:border-color .15s}
.nw:focus-within{border-color:var(--acc)}
.nw input[type=number]{flex:1;background:transparent;border:none;border-left:1px solid var(--bd2);border-right:1px solid var(--bd2);border-radius:0;text-align:center;padding:7px 4px;color:var(--tx);font-family:var(--mono);font-size:12px}
.nw input[type=number]:focus{outline:none}
.nb{width:34px;height:34px;background:transparent;border:none;color:var(--tx3);font-size:17px;cursor:pointer;transition:background .12s,color .12s;display:flex;align-items:center;justify-content:center}
.nb:hover{background:var(--sf3);color:var(--tx)}
.trow{display:flex;gap:3px}
.tbtn{flex:1;padding:6px 0;background:var(--sf2);border:1px solid var(--bd2);color:var(--tx3);font-family:var(--mono);font-size:11px;cursor:pointer;border-radius:4px;transition:color .15s,background .15s,border-color .15s}
.tbtn:hover{color:var(--tx)}
.tbtn.active{background:var(--acc);border-color:var(--acc);color:#000;font-weight:600}
.rbtn{width:100%;padding:10px;background:var(--acc);color:#000;border:none;border-radius:6px;font-family:var(--mono);font-size:12px;font-weight:700;cursor:pointer;letter-spacing:.06em;transition:opacity .2s,transform .15s,box-shadow .2s}
.rbtn:hover:not(:disabled){opacity:.88;box-shadow:0 0 16px rgba(240,96,138,.3)}
.rbtn:active:not(:disabled){transform:scale(.97)}
.rbtn:disabled{opacity:.3;cursor:not-allowed}
.gbtn{width:100%;padding:8px;background:transparent;color:var(--tx2);border:1px solid var(--bd2);border-radius:6px;font-family:var(--mono);font-size:11px;cursor:pointer;transition:border-color .15s,color .15s}
.gbtn:hover:not(:disabled){border-color:var(--acc);color:var(--tx)}
.gbtn:disabled{opacity:.4;cursor:not-allowed}
.pb-bg{height:3px;background:var(--sf3);border-radius:2px;overflow:hidden;margin-bottom:6px}
.pb-fill{height:100%;background:linear-gradient(90deg,var(--acc),#F4A0BE);border-radius:2px;width:0%;transition:width 1.4s cubic-bezier(0,.55,.45,1)}
.pb-lbl{font-family:var(--mono);font-size:10px;color:var(--tx3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lc{display:flex;flex-direction:column;gap:5px;min-height:80px;overflow:hidden}
.lcard{display:flex;align-items:flex-start;gap:8px;background:var(--sf2);border:1px solid var(--bd);border-left:2px solid var(--bd2);border-radius:5px;padding:7px 9px;will-change:transform,opacity;transform:translateY(0);opacity:1;transition:opacity .35s var(--eout),transform .35s var(--eout),border-left-color .3s,max-height .4s var(--eout),padding .4s var(--eout),margin-bottom .4s var(--eout);max-height:60px;overflow:hidden}
.lcard.ce{opacity:0;transform:translateY(-10px)}
.lcard.cx{opacity:0;transform:translateY(4px);max-height:0;padding-top:0;padding-bottom:0;margin-bottom:-5px}
.lcard.in{border-left-color:var(--acc)}.lcard.dn{border-left-color:var(--bull)}.lcard.er{border-left-color:var(--bear)}.lcard.ol{opacity:.35}
.lcard-m{font-family:var(--mono);font-size:11px;color:var(--tx2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lcard.in .lcard-m{color:var(--tx)}.lcard.dn .lcard-m{color:var(--bull)}
.lcard-t{font-family:var(--mono);font-size:9px;color:var(--tx3);margin-top:1px}
.main{display:flex;flex-direction:column;overflow:hidden;background:var(--bg);position:relative}
.rbar{display:flex;align-items:center;gap:8px;padding:9px 16px;background:var(--sf);border-bottom:1px solid var(--bd);flex-shrink:0;flex-wrap:wrap}
.rtitle{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:var(--tx3)}
.spill{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:20px;font-family:var(--mono);font-size:11px;font-weight:600;border:1px solid transparent;opacity:0;transform:scale(.85);pointer-events:none;transition:opacity .35s var(--eout),transform .35s var(--eout)}
.spill.vis{opacity:1;transform:scale(1);pointer-events:auto}
.spill .num{display:inline-block}
@keyframes nb2{0%,100%{transform:scale(1)}50%{transform:scale(1.4)}}
.num.bmp{animation:nb2 .28s var(--ease)}
.sbull{background:var(--bull-bg);color:var(--bull);border-color:#1a4020}
.sbear{background:var(--bear-bg);color:var(--bear);border-color:#4a1a22}
.sfresh{background:var(--fresh-bg);color:var(--fresh);border-color:#4a3500}
.fbar{margin-left:auto;display:flex;gap:4px;flex-wrap:wrap}
.ftab{padding:4px 10px;background:transparent;border:1px solid var(--bd2);color:var(--tx3);font-family:var(--mono);font-size:10px;font-weight:600;letter-spacing:.06em;border-radius:4px;cursor:pointer;transition:color .15s,border-color .15s,background .15s}
.ftab:hover{color:var(--tx2);border-color:var(--tx3)}
.ftab.active{background:var(--sf3);color:var(--tx);border-color:var(--acc)}
.tw{flex:1;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--bd2) transparent}
.tw::-webkit-scrollbar{width:4px}.tw::-webkit-scrollbar-track{background:transparent}.tw::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:4px}
table{width:100%;border-collapse:collapse}
thead{position:sticky;top:0;z-index:2}
th{background:var(--sf);border-bottom:1px solid var(--bd);padding:9px 12px;text-align:left;font-family:var(--mono);font-size:9px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--tx3);white-space:nowrap}
th.sortable{cursor:pointer;user-select:none}
th.sortable:hover{color:var(--tx2)}
th.sorted{color:var(--acc)}
td{padding:10px 12px;border-bottom:1px solid var(--bd);vertical-align:middle;transition:background .2s}
tr.rb{background:#0D201018}tr.rr{background:#2D0F1818}
tr.dim td{opacity:.6}
tr.dim td:nth-child(-n+3){opacity:1}
tr:hover td{background:var(--sf2)}
@keyframes ri{from{opacity:0;transform:translateX(-5px)}to{opacity:1;transform:translateX(0)}}
tbody tr{animation:ri .3s var(--eout) both}
.pb{display:flex;flex-direction:column;gap:2px}
.pb-pair{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--tx)}
.pb-q{font-family:var(--mono);font-size:9px;color:var(--tx3)}
.bdg{display:inline-flex;align-items:center;padding:3px 7px;border-radius:4px;font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.04em;white-space:nowrap}
.bb{background:var(--bull-bg);color:var(--bull);border:1px solid #1a4020}
.br{background:var(--bear-bg);color:var(--bear);border:1px solid #4a1a22}
.ptm{font-family:var(--mono);font-size:10px;color:var(--tx3)}
.sl{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;background:var(--bg);z-index:10;opacity:0;pointer-events:none;transition:opacity .35s ease}
.sl.vis{opacity:1;pointer-events:auto}.sl.diss{opacity:0;pointer-events:none}
.sbg{position:absolute;inset:0;overflow:hidden;pointer-events:none;-webkit-mask-image:linear-gradient(to bottom,#000 0%,#000 88%,transparent 100%);mask-image:linear-gradient(to bottom,#000 0%,#000 88%,transparent 100%)}
.sbg-feed{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:flex-end;padding:20px 24px;overflow:hidden}
.bgl{font-family:var(--mono);font-size:10px;line-height:1.8;color:#00C853;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex-shrink:0;opacity:0;transform:translateY(120%);transition:opacity .4s ease,transform .4s cubic-bezier(0,.55,.45,1)}
.bgl.sh{opacity:.22;transform:translateY(0)}.bgl.sh.hit{opacity:.75;color:#A8FFD0}
.lring{position:relative;width:72px;height:72px}
.lring svg{width:72px;height:72px;transform:rotate(-90deg)}
.rtrack{fill:none;stroke:var(--bd);stroke-width:2.5}
.rfill{fill:none;stroke:var(--acc);stroke-width:2.5;stroke-linecap:round;stroke-dasharray:201;stroke-dashoffset:201;transition:stroke-dashoffset 1.4s cubic-bezier(0,.55,.45,1),stroke .4s;filter:drop-shadow(0 0 4px rgba(240,96,138,.6))}
.rfill.done{stroke:var(--bull)}
.rpct{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:13px;font-weight:600;color:var(--tx)}
.lpair{font-family:var(--mono);font-size:11px;color:var(--tx2);max-width:220px;text-align:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.llbl{font-family:var(--mono);font-size:10px;color:var(--tx3);letter-spacing:.1em;text-transform:uppercase}
.qc{display:flex;flex-direction:column;gap:2px;min-width:56px}
.qnum{font-family:var(--mono);font-size:12px;font-weight:600}
.qbar{height:3px;background:var(--bd2);border-radius:2px;overflow:hidden}
.qfill{height:100%;border-radius:2px;transition:width .4s}
.tlbl{font-size:11px;font-weight:700;white-space:nowrap}
.tsub{font-size:9px;font-weight:400;opacity:.8;margin-top:1px;white-space:normal;line-height:1.3}
.ev{display:inline-flex;flex-direction:column;gap:2px;padding:6px 10px;border-radius:5px;font-family:var(--mono);min-width:220px;max-width:340px}
.ev .tsub{white-space:normal;line-height:1.45;font-size:10.5px;font-family:var(--sans);opacity:.95}
.ev-enter{background:#0D2E1F;color:#34D399;border:1px solid #1a4020}
.ev-wait{background:var(--fresh-bg);color:var(--fresh);border:1px solid #4a3500}
.ev-skip{background:var(--sf3);color:var(--tx3);border:1px solid var(--bd2)}
table.stale .ev{opacity:.4}
tbody tr.main-row{cursor:pointer}
tr.det td{background:var(--sf);font-family:var(--mono);font-size:10px;color:var(--tx2);line-height:1.8;padding:10px 16px}
tr.det b{color:var(--tx)}
tr.det img.chart{display:block;width:100%;max-width:900px;margin:8px 0;border:1px solid var(--bd2);border-radius:4px}
.tfb-on{color:var(--tx);border-color:var(--tx3)}
.tfb{display:inline-block;margin-left:4px;padding:2px 5px;border-radius:3px;font-family:var(--mono);font-size:9px;font-weight:700;background:var(--sf3);color:var(--tx2);border:1px solid var(--bd2)}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:10px;color:var(--tx3);font-family:var(--mono);font-size:12px}
.empty-icon{font-size:28px;opacity:.3}
.greet{display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center;padding:48px 40px}
.g-hl{font-family:var(--sans);font-size:42px;font-weight:600;color:var(--tx);letter-spacing:-.02em;line-height:1.15}
.g-sub{font-family:var(--sans);font-size:16px;color:var(--tx3);max-width:380px;line-height:1.7}
.g-cta{margin-top:16px;padding:14px 52px;background:var(--acc);color:#000;border:none;border-radius:7px;font-family:var(--sans);font-size:14px;font-weight:600;cursor:pointer;transition:opacity .2s,transform .15s,box-shadow .2s}
.g-cta:hover{opacity:.88;box-shadow:0 0 28px rgba(240,96,138,.4)}
.g-cta:active{transform:scale(.97)}
.dash-sec{margin-bottom:28px}
.dash-sec-ttl{font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--tx3);margin-bottom:12px}
.no-sig{color:var(--tx3);font-family:var(--mono);font-size:12px;padding:20px;background:var(--sf);border:1px solid var(--bd);border-radius:8px;text-align:center}
.audit{font-family:var(--mono);font-size:10px;color:var(--tx3);margin:-14px 0 24px;line-height:1.7}
.audit.warn{color:var(--fresh);background:var(--fresh-bg);border:1px solid #4a3500;border-radius:6px;padding:9px 12px;margin-top:-10px}
.uni-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:6px}
.uni-row{display:flex;align-items:center;justify-content:space-between;padding:6px 9px;background:var(--sf);border:1px solid var(--bd);border-radius:6px;font-family:var(--mono);font-size:11px}
.uni-base{font-weight:600;color:var(--tx)}
.uni-vol{color:var(--tx3);font-size:9px}
.learn-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.learn-card{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px}
.learn-card.lc-bull{border-left:3px solid var(--bull)}
.learn-card.lc-bear{border-left:3px solid var(--bear)}
.learn-card-hdr{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.learn-sub{font-size:11px;color:var(--tx3);font-family:var(--mono)}
.learn-panels{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.lp-lbl{font-size:9px;color:var(--tx3);text-align:center;margin-bottom:4px;letter-spacing:.08em;text-transform:uppercase}
.lp-svg{width:100%;display:block;background:var(--sf2);border-radius:5px}
.lp-line{fill:none;stroke:var(--tx3);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.lp-zero{stroke:var(--bd2);stroke-width:0.5;stroke-dasharray:2 2}
.lp-dot-A{fill:var(--sf3);stroke:var(--tx2);stroke-width:1.5}
.lp-dot-B{fill:var(--sf3);stroke:var(--acc);stroke-width:1.5}
.lp-txt{font-family:var(--mono);font-size:8px;fill:var(--tx2);text-anchor:middle}
.lp-connect-bull{stroke:var(--bull);stroke-width:1.2;stroke-dasharray:3 2}
.lp-connect-bear{stroke:var(--bear);stroke-width:1.2;stroke-dasharray:3 2}
.lp-cap{font-family:var(--mono);font-size:8px;text-anchor:middle}
.lp-cap-bull{fill:var(--bull)}
.lp-cap-bear{fill:var(--bear)}
.rules{max-width:760px}

  opacity:0;transform:translateY(8px);animation:rtIn .45s ease forwards;animation-delay:var(--d);transition:border-color .3s,background .3s,box-shadow .3s}
.rule{display:flex;gap:12px;padding:11px 0;border-bottom:1px solid var(--bd)}
.rule:last-child{border-bottom:none}
.rule-n{font-family:var(--mono);font-size:10px;color:var(--acc);flex-shrink:0;width:22px;padding-top:2px}
.rule-b{font-size:12px;color:var(--tx2);line-height:1.65}
.rule-b b{color:var(--tx);font-weight:600}
.sig-hdr{display:flex;align-items:center;gap:12px;margin-bottom:18px}
.sig-ttl{font-size:18px;font-weight:600;color:var(--tx)}
.sig-sub{font-size:12px;color:var(--tx3)}
.sig-filters{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.sig-ftab{padding:5px 12px;background:var(--sf);border:1px solid var(--bd2);color:var(--tx3);font-family:var(--mono);font-size:10px;font-weight:600;border-radius:4px;cursor:pointer;transition:all .15s}
.sig-ftab:hover{color:var(--tx2);border-color:var(--tx3)}
.sig-ftab.active{background:var(--sf3);color:var(--tx);border-color:var(--acc)}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--tx3)}


#pg-movers.active{display:block;overflow-y:auto;padding:24px}
#pg-movers{scrollbar-width:thin;scrollbar-color:var(--bd2) transparent}
.mv-thin{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--tx2);cursor:pointer;user-select:none}
.mv-thin input{accent-color:var(--fresh)}
.mv-sep td{background:var(--fresh-bg);color:var(--fresh);font-family:var(--mono);font-size:10px;letter-spacing:.06em;padding:8px 12px;border-bottom:1px solid #4a3500}
.thin-badge{display:inline-block;margin-top:3px;padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:9px;font-weight:600;background:var(--fresh-bg);color:var(--fresh);border:1px solid #4a3500;white-space:nowrap}
.mrank{display:inline-flex;align-items:center;justify-content:center;width:21px;height:21px;border-radius:5px;background:var(--sf3);border:1px solid var(--bd2);font-family:var(--mono);font-size:10px;font-weight:700;color:var(--tx3)}
.mrank.top{background:var(--acc);border-color:var(--acc);color:#000}
.mpos{position:relative;width:88px;height:5px;background:var(--sf3);border-radius:3px}
.mpos i{position:absolute;top:-3px;width:3px;height:11px;border-radius:2px;background:var(--tx);transform:translateX(-1px)}
.mlv{font-family:var(--mono);font-size:10px;color:var(--tx2);white-space:nowrap;line-height:1.6}
.mlv small{color:var(--tx3);font-size:9px}
/* ── Calendar ── */
#pg-calendar.active{display:block;overflow-y:auto;padding:24px}
#pg-calendar{scrollbar-width:thin;scrollbar-color:var(--bd2) transparent}
.cal-next{display:flex;align-items:center;gap:18px;flex-wrap:wrap;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px 18px;margin-bottom:16px}
.cal-next.soon{border-color:#4a3500;background:var(--fresh-bg)}
.cal-next-lbl{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--tx3);width:100%}
.cal-next-ttl{font-size:16px;font-weight:600;color:var(--tx)}
.cal-next-in{font-family:var(--mono);font-size:14px;font-weight:600;color:var(--acc)}
.cal-next.soon .cal-next-in{color:var(--fresh)}
.cal-next-meta{font-family:var(--mono);font-size:11px;color:var(--tx2)}
.cal-next-meta b{color:var(--tx);font-weight:600}
.cal-grid{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;margin-bottom:24px}
.cal-day{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px 10px 8px;min-height:104px;cursor:pointer;display:flex;flex-direction:column;gap:6px;transition:border-color .15s,background .15s}
.cal-day:hover{border-color:var(--bd2)}
.cal-day.past{opacity:.4}
.cal-day.today{border-color:var(--acc)}
.cal-day.soon{background:var(--fresh-bg);border-color:#4a3500}
.cal-day.sel{background:var(--sf2);box-shadow:inset 0 0 0 1px var(--tx3)}
.cal-dh{display:flex;justify-content:space-between;align-items:baseline}
.cal-dow{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:var(--tx3)}
.cal-dn{font-family:var(--mono);font-size:14px;font-weight:600;color:var(--tx)}
.cal-day.today .cal-dn{color:var(--acc)}
.cal-ev{font-size:11px;line-height:1.35;color:var(--tx);background:var(--sf3);border-left:2px solid var(--bd2);border-radius:4px;padding:5px 7px}
.cal-ev.mv{border-left-color:var(--bear)}
.cal-ev.done{opacity:.5}
.cal-ev-t{font-family:var(--mono);font-size:10px;color:var(--tx2);margin-right:5px}
.cal-ev-c{font-family:var(--mono);font-size:9px;color:var(--tx3);margin-left:4px}
.cal-quiet{font-size:11px;color:var(--tx3);font-style:italic;margin-top:auto}
.cal-imp{font-family:var(--mono);font-size:9px;font-weight:600;letter-spacing:.08em;padding:2px 6px;border-radius:3px;background:var(--sf3);color:var(--tx2);white-space:nowrap}
.cal-imp.hi{background:var(--bear-bg);color:var(--bear)}
.cal-imp.md{background:var(--fresh-bg);color:var(--fresh)}
.cal-strip{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:10px 16px;margin:0 0 24px;font-size:12px;color:var(--tx2)}
.cal-strip.soon{border-color:#4a3500;background:var(--fresh-bg)}
.cal-strip .lbl{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--tx3)}
.cal-strip b{color:var(--tx);font-weight:600}
.cal-strip .in{font-family:var(--mono);font-weight:600;color:var(--acc)}
.cal-strip.soon .in{color:var(--fresh)}
.cal-strip a{color:var(--tx3);cursor:pointer;margin-left:auto;font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase}
.cal-strip a:hover{color:var(--tx)}
.cal-out{font-family:var(--mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--tx3);text-decoration:none;border-bottom:1px solid transparent;transition:color .15s,border-color .15s;white-space:nowrap}
.cal-out:hover{color:var(--acc);border-bottom-color:var(--acc)}
.cal-day-hdr{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.cal-day-hdr .dash-sec-ttl{margin:0}
.mf-vol{background:#2D1F06;color:#FBBF24;border:1px solid #4a3500}
.mf-rng{background:#1B1030;color:#C4A0FF;border:1px solid #3A2A5A}
.mf-hi{background:var(--bull-bg);color:var(--bull);border:1px solid #1a4020}
.mf-lo{background:var(--bear-bg);color:var(--bear);border:1px solid #4a1a22}
.mf-move{background:#0F2A3A;color:#7DD3FC;border:1px solid #1F4A66}
</style>
</head>
<body>
<div class="shell">

<header class="topbar">
  <span class="t-logo">MACD&middot;DIV</span>
  <div class="t-div"></div>
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
  <aside class="sb">
    <div class="ss"><div class="fl">Exchange</div>
      <select id="exchange"><option value="okx">OKX</option><option value="binance">Binance</option><option value="bybit">Bybit</option></select></div>

    <div class="ss"><div class="fl">Minimum weekly volume ($M)</div>
      <div class="nw"><button class="nb" onclick="stepVol(-10)">-</button><input type="number" id="minVol" value="__DEFAULT_VOL_M__" min="5" max="2000" step="10"><button class="nb" onclick="stepVol(10)">+</button></div>
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
      <div class="nw"><button class="nb" onclick="stepPairs(-25)">-</button><input type="number" id="maxPairs" value="__DEFAULT_MAX_PAIRS__" min="10" max="600" step="25"><button class="nb" onclick="stepPairs(25)">+</button></div>
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
      <div class="hint">Comma-separated. Overrides the volume filter entirely &mdash; scans exactly these.</div></div>

    <div class="ss"><button class="rbtn" id="runBtn" onclick="startScan()">Run Scan</button></div>
    <div class="ss" id="pgSec" style="display:none">
      <div class="fl">Progress</div>
      <div class="pb-bg"><div class="pb-fill" id="pbFill"></div></div>
      <div class="pb-lbl" id="pbLbl">-</div></div>
    <div class="ss" style="flex:1"><div class="fl">Activity</div><div class="lc" id="lc"></div></div>
  </aside>

  <main class="main">
    <div class="sl" id="sl">
      <div class="sbg"><div class="sbg-feed" id="sbgFeed"></div></div>
      <div class="lring">
        <svg viewBox="0 0 72 72"><circle class="rtrack" cx="36" cy="36" r="32"/><circle class="rfill" id="rfill" cx="36" cy="36" r="32"/></svg>
        <div class="rpct" id="rpct">0%</div>
      </div>
      <div class="lpair" id="lpair">Loading markets...</div>
      <div class="llbl">Scanning for divergences</div>
    </div>
    <div class="cal-strip" id="dash-cal" style="display:none;margin:10px 16px 0"></div>
    <div class="rbar">
      <span class="rtitle">Divergences</span>
      <span class="spill sbull" id="stBull">&#9650; <span class="num" id="cBull">0</span> bullish</span>
      <span class="spill sbear" id="stBear">&#9660; <span class="num" id="cBear">0</span> bearish</span>
      <span class="spill sfresh" id="stEnter">&#9889; <span class="num" id="cEnter">0</span> enter now</span>
      <div class="fbar">
        <button class="ftab" data-f="live" onclick="setF(this)" title="Everything still worth watching: verdict Enter or Wait.">Live</button>
        <button class="ftab" data-f="enter" onclick="setF(this)" title="Only rows whose verdict is Enter.">Enter now</button>
        <button class="ftab" data-f="bull" onclick="setF(this)">Bullish</button>
        <button class="ftab" data-f="bear" onclick="setF(this)">Bearish</button>
        <button class="ftab" data-f="both" onclick="setF(this)" title="The same side diverging on 4H and 1H at once.">4H+1H</button>
        <button class="ftab active" data-f="all" onclick="setF(this)" title="Every divergence found, including ones that already played out, were stopped, or are too old.">All</button>
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
          <th class="sortable sorted" data-sort="score" title="A checklist, not a prediction. First matching rule wins. No: the stop was hit, price already ran 1R, the signal bar is older than 6 bars on 4H or 12 on 1H, or the grade is C. Not yet: counter trend below grade A, price more than 2% past the signal price, a market-mover print inside 24 hours, or a signal bar still forming with no lower-timeframe trigger. Otherwise Yes. Default order: Yes, then Not yet, then No; highest score first, No rows newest first.">Good to enter? &#8595;</th>
          <th title="Regular bullish: across two MACD troughs, price makes a lower low while the MACD line makes a higher trough. Regular bearish is the mirror on two peaks. The tags show which timeframes diverge; the highlighted one supplies the answer and the levels.">Side &middot; TF</th>
          <th title="Quality score 0-100 and its grade: A is 70 or more, B is 55-69, C is below 55. Pattern points (up to 70) plus context points (higher timeframe, OI with funding, BTC state: 10 each). Context the data cannot check is left out and the score is rescaled to 100 (tag PARTIAL DATA). It describes the setup; it is not a win rate. Run the backtest to see how each grade has done.">Grade</th>
          <th title="WITH TREND: the BTC 4H state (BULL or BEAR) and this coin's 200 EMA both agree with the side. CHOP: BTC is neither BULL nor BEAR; a reversal keeps the normal cutoff. COUNTER TREND (also when BTC data is missing) needs grade A to answer Yes. Below it: what price has done since the signal bar.">Regime &middot; Status</th>
          <th class="sortable" data-sort="age" title="The signal bar is the MACD cross that ends the second wave. Shown: its closing price, and the bar's time as a chart labels it (the bar's open).">Signal</th>
          <th title="Beyond price extreme B by 0.25 ATR. Price trading through it means the divergence has failed.">Stop</th>
          <th title="Multiples of the distance from the entry reference to the stop. The entry reference is the signal price, or the lower-timeframe trigger price when that fired first. The structure target is the price at the top of the bump between A and B.">Targets (1R / 2R / 3R)</th>
          <th title="Times of the price extremes inside wave A and wave B.">A &rarr; B</th>
          <th class="sortable" data-sort="oi" title="Change in this coin's open interest (in dollars) over the last 7 days. Context only: it never changes the answer or the score. n/a means the exchange has too little OI history for this contract. Click to sort, biggest build-up first.">7d OI</th>
          <th title="Pattern tags first (TRIPLE, W-PATTERN, LTF TRIGGER, ZERO-LINE CROSS): TRIPLE is worth 8 score points and LTF TRIGGER is what allows Yes (early). PARTIAL DATA means some score context could not be checked. Then the daily-movers and open-interest tags, which are context only and never change the answer. Hover a tag for its definition, or open the Tags page for the full list.">Tags</th>
        </tr></thead>
        <tbody id="resB"></tbody>
      </table>
    </div>
  </main>
</div>

<div id="pg-movers" class="page">
  <div class="sig-hdr">
    <div><div class="sig-ttl">Movers &mdash; daily watchlist</div>
    <div class="sig-sub" id="mv-sub">Unusual volume, unusual range, or a big day &mdash; measured on completed daily bars. Crypto perps only. Uses the Scan tab's volume floor, speed and max-pairs.</div></div>
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
  <div id="mv-audit" class="audit" style="margin:0 0 12px"></div>
  <div class="tw" style="overflow:visible">
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
    <div class="sig-filters" style="margin:0">
      <button class="sig-ftab active" data-cf="movers" onclick="setCF(this)" title="Only the curated list: CPI, PCE, PPI, the jobs report, FOMC, Powell, GDP, retail sales, ISM, and the BoJ / ECB / BoE rate decisions.">Market movers</button>
      <button class="sig-ftab" data-cf="high" onclick="setCF(this)" title="Everything ForexFactory rates High impact, all currencies.">All high-impact</button>
    </div>
    <button class="rbtn" id="calRefreshBtn" onclick="refreshCalendar()">Refresh</button>
    <span class="wprog" id="cal-meta"></span>
    <a class="cal-out" style="margin-left:auto" href="https://www.forexfactory.com/calendar" target="_blank" rel="noopener"
       title="Open ForexFactory's calendar in a new tab. The feed carries no actual values, so this is where you check what a print came in at.">ForexFactory &#8599;</a>
  </div>
  <div id="cal-audit" class="audit" style="margin:0 0 12px"></div>
  <div class="cal-grid" id="cal-grid"></div>
  <div class="dash-sec">
    <div class="cal-day-hdr">
      <div class="dash-sec-ttl" id="cal-day-ttl">Select a day</div>
      <a class="cal-out" id="cal-ff-day" href="https://www.forexfactory.com/calendar" target="_blank" rel="noopener"
         title="Open this day on ForexFactory, where the actual values appear once a print lands.">Open this day on ForexFactory &#8599;</a>
    </div>
    <div class="tw" style="overflow:visible">
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

<div id="pg-tags" class="page" style="overflow-y:auto;padding:24px">
  <div class="dash-sec">
    <div class="dash-sec-ttl">What every tag means</div>
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75">
      Each tag as it appears in the table, with the same wording as its hover text. Only the pattern
      tags TRIPLE and LTF TRIGGER feed the answer; every other tag is context for you to weigh.
    </div>
  </div>
  <div id="tagsBody"></div>
</div>

<div id="pg-learn" class="page" style="overflow-y:auto;padding:24px">
  <div class="dash-sec">
    <div class="dash-sec-ttl">What this screener looks for</div>
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75;margin-bottom:18px">
      Regular divergence only &mdash; price and momentum disagreeing at a trend extreme, which is the
      classic <b style="color:var(--tx)">reversal</b> signature. Hidden (continuation) divergences are
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

  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">Good to enter? &mdash; a checklist, not a prediction</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>No comes first.</b> The stop was hit, price already made 1R or more, the signal bar is older than 6 bars on 4H or 12 on 1H, or the grade is C.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Not yet names the one thing missing.</b> A counter-trend setup below grade A (watch only), a pullback to the signal price after price ran more than 2%, a CPI / FOMC-class print inside 24 hours, or a signal bar that has not closed and has no lower-timeframe trigger.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Yes (confirmed) and Yes (early).</b> Confirmed means the signal bar has closed and nothing above applies. Early means the signal bar is still forming but the lower timeframe (1H for 4H, 15m for 1H) has already crossed its MACD and closed beyond its last swing since B.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Regime changes the bar, it never blocks.</b> BTC 4H is BULL (RSI above 55 and close above its 200 EMA), BEAR (RSI below 45 and close below it) or CHOP (anything else). WITH TREND means the BTC state and the coin's own 200 EMA both agree with the signal. CHOP keeps the normal cutoff, because ranges are where reversals work. Anything else, missing BTC data included, is COUNTER TREND and needs grade A.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>The answer ages.</b> It uses the live price at the moment of the scan. Once the scan is more than an hour old the answers dim and the page asks for a re-run.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>One row per coin and side.</b> When 4H and 1H diverge the same way the row shows both tags and takes its answer, grade and levels from the better one (Yes beats Not yet beats No, 4H wins a tie); the highlighted tag shows which. Click a row for each timeframe's levels and its chart.</div></div>
    </div>
  </div>

  <div class="dash-sec" style="margin-top:28px">
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
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75">
      0 to 100, graded <b style="color:var(--tx)">A</b> (70+), <b style="color:var(--tx)">B</b> (55-69) or
      <b style="color:var(--tx)">C</b> (below 55). Pattern points, up to 70: the price difference A to B in ATR
      (12), the MACD trough improvement (12), the height of the bump (8), both troughs on the classic side of
      zero (10), volume drying up in wave B (10), a rejection wick at B (10), a triple divergence (8). Context
      points, 10 each: the higher timeframe's MACD agreeing, open interest falling into B with funding not
      against it, and the BTC 4H state on the signal's side (BULL for bullish, BEAR for bearish; CHOP earns
      nothing). Open interest rising more than 5% against the signal costs 10.
      <br><br>
      A context item the data cannot check (no higher-timeframe candles, no OI history or funding, no BTC
      candles) is left out rather than scored as a miss: the score is what was earned out of what could be
      checked, scaled to 100, and the signal is tagged <b style="color:var(--tx)">PARTIAL DATA</b>.
      <br><br>
      The grade describes the setup. It is not a win rate. To see how each grade and regime has actually done on
      the stored candles, run <b style="color:var(--tx)">python macd_divergence_screener_v5.py --backtest</b>.
    </div>
  </div>
</div>

</div>

<script>
var TFM = __TF_CFG_JSON__;

var SC = {
  active:      {lbl:'At the signal'},
  extended:    {lbl:'Extended'},
  ran:         {lbl:'Ran'},
  invalidated: {lbl:'Invalidated'}
};

function qcolor(q){ return q >= 70 ? 'var(--bull)' : q >= 50 ? 'var(--fresh)' : 'var(--tx3)'; }
function qcell(q){
  var c = qcolor(q), lbl = q >= 70 ? 'HIGH' : q >= 50 ? 'MED' : 'LOW';
  return '<div class="qc"><span class="qnum" style="color:' + c + '">' + q +
    '<span style="font-size:9px;opacity:.7;margin-left:3px">' + lbl + '</span></span>' +
    '<div class="qbar"><div class="qfill" style="width:' + q + '%;background:' + c + '"></div></div></div>';
}
function gradeCell(p){
  var c = p.grade === 'A' ? 'var(--bull)' : p.grade === 'B' ? 'var(--fresh)' : 'var(--tx3)';
  return '<div class="qc"><span class="qnum" style="color:' + c + '">' + p.grade +
    '<span style="font-size:9px;opacity:.7;margin-left:3px">' + p.score + '</span></span>' +
    '<div class="qbar"><div class="qfill" style="width:' + p.score + '%;background:' + c + '"></div></div></div>';
}
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

var EV = {
  enter: {lbl:'&#10003; YES',     cls:'ev-enter'},
  wait:  {lbl:'&#9203; NOT YET', cls:'ev-wait'},
  skip:  {lbl:'&#10005; NO',     cls:'ev-skip'}
};
var EV_RANK = {enter:0, wait:1, skip:2};
function verdict(r){ return (r.entry && r.entry.verdict) || 'skip'; }
function primary(r){ return r.signals[r.primary_tf || r.tfs[0]]; }

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
  var h = [];
  (primary(r).tags || []).forEach(function(f){
    h.push('<span class="wflag wf-hioi"' + tip(f) + '>' + f + '</span>');
  });
  if(r.context_error) h.push('<span class="ptm" title="' + calEsc(r.context_error) + '">context unavailable</span>');
  ((r.mover || {}).flags || []).forEach(function(f){
    h.push('<span class="wflag ' + (MFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>');
  });
  if(r.flow && r.flow.quadrant){
    h.push('<span class="' + (WQ[r.flow.quadrant] || {cls:'wq'}).cls + '"' + tip(r.flow.quadrant) + '>' + r.flow.quadrant + '</span>');
    (r.flow.flags || []).forEach(function(f){
      h.push('<span class="wflag ' + (WFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>');
    });
  }
  return h.length ? '<div style="display:flex;flex-wrap:wrap;gap:4px;max-width:260px">' + h.join('') + '</div>'
                  : '<span class="ptm">&mdash;</span>';
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

/* ── OI flow tags ─────────────────────────────────────────────────────────── */
var WQ = {
  'NEW LONGS':      {cls:'bdg bb'},
  'NEW SHORTS':     {cls:'bdg br'},
  'SHORT COVERING': {cls:'wq wq-sc'},
  'CAPITULATION':   {cls:'wq wq-cap'}
};
var WFLAG = {'SPOT-LED':'wf-spot','CROWDED LONGS':'wf-crowd','SQUEEZE FUEL':'wf-squeeze','HIGH OI/VOL':'wf-hioi'};

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
  'ZERO-LINE CROSS': 'The MACD line has already crossed zero since the signal bar.',
  'WITH TREND':      'The BTC 4H state agrees with the signal (BULL: RSI above 55 and close above its 200 EMA; BEAR: RSI below 45 and close below it) and so does the coin\'s own 200 EMA. Grade B is enough for Yes.',
  'COUNTER TREND':   'BTC (BULL or BEAR) or the coin\'s 200 EMA disagrees with the signal, or BTC data is missing. Needs grade A for Yes; otherwise watch only.',
  'CHOP':            'BTC 4H is neither BULL nor BEAR: RSI between 45 and 55, or RSI and the 200 EMA disagree. A reversal keeps the normal cutoff (grade B is enough for Yes); the BTC score points are not earned.',
  'PARTIAL DATA':    'Some score context could not be checked (higher timeframe, open interest or funding, or BTC). It is left out and the score is what was earned out of what could be checked, scaled to 100.',
  '7d OI':           'Change in the coin\'s open interest over the last 7 days. Rising = positions being opened, falling = positions being closed. Read it with price: the four positioning tags above are exactly that combination. n/a = too little OI history.'
};
function tip(k){ return TIP[k] ? ' title="' + TIP[k] + '"' : ''; }

/* ── Movers tags ──────────────────────────────────────────────────────────── */
var MFLAG = {'VOLUME SPIKE':'mf-vol','RANGE EXPANSION':'mf-rng','BIG MOVE':'mf-move',
             'AT RANGE HIGH':'mf-hi','ABOVE 7D HIGH':'mf-hi',
             'AT RANGE LOW':'mf-lo','BELOW 7D LOW':'mf-lo'};

/* ── Tags page: built from TIP so it cannot drift from the hover text ─────── */
var TAG_GROUPS = [
  ['Pattern tags (about the divergence itself)', ['TRIPLE','W-PATTERN','LTF TRIGGER','ZERO-LINE CROSS']],
  ['Regime (sets how high the grade must be)', ['WITH TREND','CHOP','COUNTER TREND']],
  ['Score', ['PARTIAL DATA']],
  ['Open interest: 7-day positioning', ['7d OI','NEW LONGS','NEW SHORTS','SHORT COVERING','CAPITULATION']],
  ['Open interest: funding and crowding', ['SPOT-LED','CROWDED LONGS','SQUEEZE FUEL','HIGH OI/VOL']],
  ['Daily movers (the last completed day)', ['VOLUME SPIKE','RANGE EXPANSION','BIG MOVE','AT RANGE HIGH','ABOVE 7D HIGH','AT RANGE LOW','BELOW 7D LOW']]
];
function tagChip(k){
  var cls = WQ[k] ? WQ[k].cls : 'wflag ' + (MFLAG[k] || WFLAG[k] || 'wf-hioi');
  return '<span class="' + cls + '">' + k + '</span>';
}
document.getElementById('tagsBody').innerHTML = TAG_GROUPS.map(function(g){
  return '<div class="dash-sec" style="margin-top:28px"><div class="dash-sec-ttl">' + g[0] + '</div><div class="rules">' +
    g[1].map(function(k){
      return '<div class="rule"><div style="flex-shrink:0;width:150px">' + tagChip(k) + '</div><div class="rule-b">' + TIP[k] + '</div></div>';
    }).join('') + '</div></div>';
}).join('');
function oiVal(r){ return r.flow && r.flow.oi_chg_pct != null ? r.flow.oi_chg_pct : null; }
function oiCell(r){
  var v = oiVal(r);
  if(v == null) return '<span class="ptm">n/a</span>';
  return '<span style="font-family:var(--mono);font-size:11px;color:' +
    (v >= 5 ? 'var(--bull)' : v <= -5 ? 'var(--bear)' : 'var(--tx3)') + '">' + (v > 0 ? '+' : '') + v + '%</span>';
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
  var flags = (r.flags || []).map(function(f){
    return '<span class="wflag ' + (MFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>';
  }).join('') || '<span style="color:var(--tx3)">&mdash;</span>';
  var up  = r.chg_1d_pct >= 0;
  var big = Math.abs(r.chg_1d_pct || 0) >= TH.chg;
  var badge = thin ? '<div><span class="thin-badge" title="Below the volume floor. Thin book: expect worse fills and slippage through stops. Size down.">THIN &middot; ' +
                     fmtVol(r.daily_volume) + '/day</span></div>' : '';
  var tr = document.createElement('tr');
  tr.className = up ? 'rb' : 'rr';
  tr.innerHTML =
    '<td><span class="mrank' + (!thin && i < 4 ? ' top' : '') + '">' + (i + 1) + '</span></td>' +
    '<td><div class="pb"><span class="pb-pair">' + r.base + '</span><span class="pb-q">' + fmtPx(r.current_price) + '</span></div>' + badge + '</td>' +
    '<td>' + qcell(r.score) + '</td>' +
    '<td><span style="font-family:var(--mono);font-weight:600;color:' + (big ? 'var(--fresh)' : up ? 'var(--bull)' : 'var(--bear)') + '">' + fmtPct(r.chg_1d_pct) + '</span>' +
      '<div style="font-family:var(--mono);font-size:9px;color:var(--tx3)">today ' + fmtPct(r.chg_live_pct) + '</div></td>' +
    '<td><span style="font-family:var(--mono);font-size:12px;color:' + ((r.vol_mult || 0) >= TH.vol ? 'var(--fresh)' : 'var(--tx2)') + '">' + fmtX(r.vol_mult) + '</span></td>' +
    '<td><span style="font-family:var(--mono);font-size:12px;color:' + ((r.range_mult || 0) >= TH.rng ? 'var(--fresh)' : 'var(--tx2)') + '">' + fmtX(r.range_mult) + '</span></td>' +
    '<td><div class="mpos"><i style="left:' + r.pos_in_range + '%"></i></div>' +
      '<div style="font-family:var(--mono);font-size:9px;color:var(--tx3);margin-top:3px">' + r.pos_in_range + '% of range</div></td>' +
    '<td><div class="mlv"><small>PDH</small> ' + fmtPx(r.pdh) + ' &nbsp;<small>PDL</small> ' + fmtPx(r.pdl) + '</div>' +
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

/* ── Sidebar controls ─────────────────────────────────────────────────────── */
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
      out.innerHTML = '<span style="color:var(--bull)">' + d.passing + '</span> of ' + d.total_perps +
        ' perps clear this floor on ' + (d.exchange || '').toUpperCase() + '.';
    })
    .catch(function(){ btn.disabled = false; btn.textContent = 'How many coins pass?'; out.textContent = 'Failed.'; });
}

/* ── Scanner ──────────────────────────────────────────────────────────────── */
var cb = 0, cr = 0, cf = 0;
function bump(el, v){ el.textContent = v; el.classList.remove('bmp'); void el.offsetWidth; el.classList.add('bmp'); }
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
function setF(btn){
  document.querySelectorAll('.ftab').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active'); af = btn.dataset.f;
  if(hasScanned) renderAll(pendR);   // before a scan, leave the welcome panel alone
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
  d.innerHTML = '<div style="flex:1;min-width:0"><div class="lcard-m">' + msg + '</div><div class="lcard-t">' + t + '</div></div>';
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
  document.getElementById('rfill').classList.remove('done');
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
      (s.partial ? ' (partial data)' : '') +
      ' &nbsp;&middot;&nbsp; HTF ' + (s.htf_ok == null ? 'n/a' : s.htf_ok ? 'agrees' : 'no') + ', OI ' + (s.oi_chg_pct == null ? 'n/a' : s.oi_chg_pct + '%') +
      ', BTC ' + (s.btc_state || 'n/a') +
      ', volume ' + (s.vol_dry ? 'drying' : 'not drying') + ', wick ' + (s.wick ? 'yes' : 'no') + (s.triple ? ', triple' : '') +
      ' &nbsp;&middot;&nbsp; ' + c.lbl + ' (' + (s.pct_chg >= 0 ? '+' : '') + s.pct_chg + '% in signal direction)' +
      (s.trigger_price != null ? ' &nbsp;&middot;&nbsp; lower-timeframe trigger at ' + fmtPx(s.trigger_price) : '') +
      ' &nbsp;&middot;&nbsp; stop ' + fmtPx(s.stop_price) + ' (risk ' + s.risk_pct + '%)' +
      ' &nbsp;&middot;&nbsp; 1R ' + fmtPx((s.targets || {})['1r']) + ' / 2R ' + fmtPx((s.targets || {})['2r']) +
      ' / 3R ' + fmtPx((s.targets || {})['3r']) + '</div>' +
      '<img class="chart" loading="lazy" alt="Chart needs a fresh scan" src="/chart?pair=' +
      encodeURIComponent(r.pair) + '&side=' + r.side + '&tf=' + tf + '">';
  }).join('');
  tr.innerHTML = '<td colspan="11"><div>Weekly volume ' + fmtVol(r.weekly_volume) + ' (7 days)</div>' + h + '</td>';
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
    '<td>' + oiCell(r) + '</td>' +
    '<td>' + ctxCell(r) + '</td>';
  tr.addEventListener('click', function(){
    var n = tr.nextSibling;
    if(n && n.className === 'det'){ n.parentNode.removeChild(n); return; }
    tr.parentNode.insertBefore(detailRow(r), tr.nextSibling);
  });
  return tr;
}
function ageMin(r){ var tf = r.primary_tf || r.tfs[0]; return primary(r).bars_ago * ((TFM[tf] && TFM[tf].minutes) || 240); }

function renderAll(results){
  var rows = (results || []).filter(function(d){ return matches(d, af); });
  rows.sort(function(a, b){
    if(sortKey === 'age') return ageMin(a) - ageMin(b);
    if(sortKey === 'oi'){ var x = oiVal(a), y = oiVal(b); return (y == null ? -1e9 : y) - (x == null ? -1e9 : x); }
    return (EV_RANK[verdict(a)] - EV_RANK[verdict(b)]) ||
           (verdict(a) === 'skip' ? ageMin(a) - ageMin(b) : 0) ||   /* played-out rows: newest first */
           (b.score - a.score);
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
    var N = (results || []).length;
    em.innerHTML = '<span class="empty-icon">&#9676;</span><span>' + (N > 0 && af !== 'all'
      ? 'No live setups right now. ' + N + ' divergence' + (N !== 1 ? 's' : '') + ' found, all played out, stopped or too old: see All.'
      : 'No divergences match this filter') + '</span>';
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

var es = null;
function startScan(){
  showPg('scan');
  if(es){ es.close(); es = null; }
  resetStats();
  document.getElementById('lc').querySelectorAll('.lcard').forEach(function(c){
    c.classList.add('cx');
    setTimeout(function(){ if(c.parentNode) c.parentNode.removeChild(c); }, 420);
  });
  document.getElementById('resB').innerHTML = '';
  document.getElementById('resT').style.display = 'none';
  document.getElementById('emS').style.display = 'flex';
  document.getElementById('emS').innerHTML = '<span class="empty-icon">&#9672;</span><span>Scanning...</span>';
  document.getElementById('pgSec').style.display = '';
  document.getElementById('runBtn').disabled = true;
  tot = 0; setProgress(0, 0, ''); setStatus('Scanning...', true);
  showLoader(); resetBg(); pendR = []; seenSig = {}; hasScanned = true;
  scanTs = null; updAge();
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
    scanTs = Date.now(); updAge();
    document.getElementById('scan-audit').innerHTML = auditLine(m);
    if(m.vol_rejected) pushLog(m.vol_rejected + ' pair(s) fell short on 7-day volume', 'in');
    if(m.no_data || m.errors) pushLog(((m.no_data||0) + (m.errors||0)) + ' pair(s) could not be analysed (no candles / error)', 'er');
    setRing(100, '');
    document.getElementById('rfill').classList.add('done');
    if(d.hits === 0){
      setTimeout(function(){ dissolve(); }, 600);
      document.getElementById('emS').style.display = 'flex';
      document.getElementById('emS').innerHTML = '<span class="empty-icon">&#9676;</span><span>No divergences found</span>';
    } else {
      setTimeout(function(){
        dissolve();
        setTimeout(function(){ renderAll(pendR); }, 350);
      }, 500);
    }
  });
  es.addEventListener('error', function(e){
    if(!e.data) return;  // native connection errors handled by es.onerror
    var m = 'Scan error';
    try { m = JSON.parse(e.data).msg; } catch(x){}
    pushLog(m, 'er'); es.close();
    document.getElementById('runBtn').disabled = false;
    setStatus('Error', false);
  });
  es.onerror = function(){
    if(es.readyState === EventSource.CLOSED) return;
    // EventSource auto-reconnects on any transport drop, and a reconnect here
    // re-issues GET /scan - a second full market scan, sharing the exchange
    // quota with the first and appending duplicate rows to pendR. Close it and
    // let the user decide to re-run.
    es.close();
    pushLog('Connection lost - scan stopped. Press Run Scan to restart.', 'er');
    document.getElementById('runBtn').disabled = false;
    setStatus('Disconnected', false);
    if(pendR.length){ scanTs = Date.now(); updAge(); dissolve(); setTimeout(function(){ renderAll(pendR); }, 350); }
  };
}

function renderWelcome(){
  try {
    var el = document.getElementById('emS');
    if(!el) return;
    var h = new Date().getHours();
    var gr = h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
    var subs = ['Ready to find divergences.',
                '4H and 1H, regular divergences only.',
                'Every liquid perp, one scan, one table.',
                'Set the volume floor and hit Run.'];
    el.style.display = 'flex';
    el.innerHTML = '<div class="greet"><div class="g-hl">' + gr + ', Kevin</div><div class="g-sub">' +
      subs[Math.floor(Math.random()*subs.length)] + '</div><button class="g-cta" onclick="startScan()">Run Scan</button></div>';
  } catch(e){}
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
    var impCls = e.impact === 'High' ? ' hi' : (e.impact === 'Medium' ? ' md' : '');
    tr.innerHTML =
      '<td><span style="font-family:var(--mono)" title="' + calUTC(t) + '">' + calHM(t) + '</span></td>' +
      '<td><span style="font-family:var(--mono);color:var(--tx2)">' + calEsc(e.currency) + '</span></td>' +
      '<td><span style="font-weight:600;color:var(--tx)">' + calEsc(e.title) + '</span>' +
        (e.mover ? ' &nbsp;<span class="cal-imp hi" title="On the curated market-mover list">MOVER</span>' : '') + '</td>' +
      '<td><span class="cal-imp' + impCls + '">' + calEsc(e.impact) + '</span></td>' +
      '<td style="font-family:var(--mono)">' + (calEsc(e.forecast) || '&mdash;') + '</td>' +
      '<td style="font-family:var(--mono)">' + (calEsc(e.previous) || '&mdash;') + '</td>' +
      '<td style="font-family:var(--mono);color:var(--tx)">' + (calEsc(e.actual) || '&mdash;') + '</td>';
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
  a.style.margin = '0 0 12px';
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
