#!/usr/bin/env python3
"""MACD Divergence Screener v4.0 — full-market · regular divergences only

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
   removes one API call per pair — a third of the old scan cost). Entry
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
minimum divergence. LOOSE drops the MACD-pivot-alignment and clean-line
requirements — more signals, more junk.

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
import ccxt, pandas as pd, numpy as np, json, time, threading, queue, datetime, os
import re, statistics, urllib.request, urllib.error

app = Flask(__name__)

# ── Indicator constants ────────────────────────────────────────────────────────
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ATR_PERIOD  = 14
PIVOT_SYNC  = 3      # bars of slack allowed between a price pivot and its MACD pivot

# Per-timeframe geometry. Pivot windows that make sense on 4H are far too wide
# on the daily, so every distance is defined per timeframe rather than shared.
TF_CFG = {
    "4h": {"label": "4H",  "minutes": 240,  "bars": 720, "pivot_lb": 5,
           "min_dist": 5, "max_dist": 120, "max_age": 60,  "fresh": 12},
    "1d": {"label": "1D",  "minutes": 1440, "bars": 500, "pivot_lb": 5,
           "min_dist": 4, "max_dist": 90,  "max_age": 30,  "fresh": 3},
}

# Strictness presets — see the module docstring for what each rule does.
STRICTNESS = {
    "loose":    {"require_macd_pivot": False, "require_clean": False,
                 "min_price_pct": 0.15, "atr_frac": 0.00, "min_macd_frac": 0.00,
                 "zero_line": False},
    "balanced": {"require_macd_pivot": True,  "require_clean": True,
                 "min_price_pct": 0.25, "atr_frac": 0.20, "min_macd_frac": 0.08,
                 "zero_line": False},
    "strict":   {"require_macd_pivot": True,  "require_clean": True,
                 "min_price_pct": 0.50, "atr_frac": 0.55, "min_macd_frac": 0.22,
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
MOVED_PCT              = 2.0             # beyond this, re-anchor targets to spot
DISPLAY_TZ_OFFSET_H    = 8               # PHT / UTC+8 for displayed timestamps

# Threading: workers and requests/sec per exchange. OKX's candle endpoints are
# the tightest of the three, so it gets its own (lower) numbers.
SPEED = {
    "safe":   {"workers": 3,  "rps": {"okx": 4,  "binance": 6,  "bybit": 6}},
    "normal": {"workers": 6,  "rps": {"okx": 8,  "binance": 14, "bybit": 14}},
    "fast":   {"workers": 10, "rps": {"okx": 14, "binance": 22, "bybit": 22}},
}

# ── Watchlist (OI flow) ────────────────────────────────────────────────────────
# The "check OI on Sunday" routine, mechanised. A pair makes the watchlist when
# its open interest moved meaningfully over the lookback; direction of price and
# OI together give the quadrant, and funding says whether the move is spot-led
# or leveraged-crowded. Thresholds are heuristics, stated here once:
OI_LOOKBACK_D      = 7      # compare OI now vs ~7 days ago
OI_MIN_CHANGE_PCT  = 5.0    # below this the pair is "quiet" and not listed
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

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scan_state.json")
# Own file on purpose: a movers scan must never be able to clobber the
# divergence scan's saved state, and vice versa.
MOVERS_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "movers_state.json")

# ── News calendar ──────────────────────────────────────────────────────────────
# Scheduled macro prints, so a setup is never sized into a CPI or FOMC release
# by accident. JBlanked (ForexFactory's data behind a free API key) is the
# primary source; ForexFactory's own weekly JSON is the keyless fallback.
CAL_STATE_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "calendar_state.json")
CAL_REFRESH_MIN   = 30    # minutes between upstream fetches
CAL_MIN_REFRESH_S = 300   # floor for a manual Refresh: JBlanked's free tier allows one call per 5 min
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
_state = {"results": [], "ts": None, "meta": {},
          "watch": {"rows": [], "ts": None, "meta": {}}}
_lock  = threading.Lock()
_movers_state = {"rows": [], "ts": None, "meta": {}}
_cal_state = {"events": [], "ts": None, "attempt": None, "meta": {}}


# ── Indicators ─────────────────────────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()


def compute_indicators(df):
    """MACD line / signal / histogram plus ATR14 (Wilder), on completed bars."""
    close = df["close"]
    ml = ema(close, MACD_FAST) - ema(close, MACD_SLOW)
    sl = ema(ml, MACD_SIG)

    prev = close.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev).abs(),
                    (df["low"] - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False).mean()

    return pd.DataFrame({"macd": ml, "signal": sl, "hist": ml - sl, "atr": atr},
                        index=df.index)


# ── Pivots ─────────────────────────────────────────────────────────────────────
def find_pivots(vals, lb):
    """Indices that are the extreme of a 2*lb+1 window centred on them.

    Ties resolve to the EARLIEST bar (np.argmax/argmin return the first
    occurrence), so a flat double-bottom produces exactly one pivot instead of
    zero. v3 required a strictly unique extreme and silently dropped every
    flat-bottomed swing — common on low-tick-size alts.

    A pivot needs lb bars on BOTH sides, so the newest possible pivot is lb
    bars old. That is what makes this non-repainting: a pivot never appears,
    moves, or disappears on a later scan.
    """
    n = len(vals)
    highs, lows = [], []
    for i in range(lb, n - lb):
        w = vals[i - lb:i + lb + 1]
        if i - lb + int(np.argmax(w)) == i:
            highs.append(i)
        if i - lb + int(np.argmin(w)) == i:
            lows.append(i)
    return highs, lows


def swing_pivots(high, low, lb):
    """Price pivots on true swing extremes: highs from `high`, lows from `low`.

    v3 ran both off the close, which puts the pivot on the wrong bar whenever
    the actual extreme was a wick — and wicks are exactly where divergences get
    drawn from.
    """
    ph, _ = find_pivots(high, lb)
    _, pl = find_pivots(low, lb)
    return ph, pl


# ── Divergence detection ───────────────────────────────────────────────────────
def _macd_anchor(macd, macd_piv_set, idx, mode, need_pivot):
    """Find the MACD swing that belongs to the price pivot at `idx`.

    Returns (bar, value) for the MACD pivot within PIVOT_SYNC bars — the
    lowest one for a price trough, the highest for a price peak. Momentum
    commonly turns a bar or two before or after price, so demanding they land
    on the same bar throws away good setups; demanding nothing accepts bars
    where MACD was not swinging at all.

    In loose mode (need_pivot=False) this falls back to the MACD value at the
    price pivot bar, which is what v3 always did.
    """
    cands = [j for j in macd_piv_set if abs(j - idx) <= PIVOT_SYNC]
    if not cands:
        if need_pivot:
            return None
        return idx, float(macd[idx])
    j = min(cands, key=lambda k: macd[k]) if mode == "low" else max(cands, key=lambda k: macd[k])
    return j, float(macd[j])


def detect_divergences(df, cfg, rules):
    """REGULAR bullish / bearish divergences only.

    Bullish : price makes a LOWER LOW, MACD makes a HIGHER LOW  → reversal up
    Bearish : price makes a HIGHER HIGH, MACD makes a LOWER HIGH → reversal down

    Every candidate right-hand pivot B is tested against ALL prior pivots A in
    the [min_dist, max_dist] window (v3.2's fix — the nearest prior pivot is
    usually a shallow intermediate swing that does not diverge, so pairing only
    with it missed most chart-visible divergences). Among the A's that survive
    every rule, the one with the deepest MACD extreme wins: that is the line a
    human would draw.
    """
    high = df["high"].values
    low  = df["low"].values
    close = df["close"].values
    macd = df["macd"].values
    atrv = df["atr"].values
    n = len(df)

    lb  = cfg["pivot_lb"]
    mn, mx, age = cfg["min_dist"], cfg["max_dist"], cfg["max_age"]

    price_h, price_l = swing_pivots(high, low, lb)
    macd_h,  macd_l  = find_pivots(macd, lb)
    macd_h, macd_l = set(macd_h), set(macd_l)

    out = []

    # ── Regular bullish: lower low in price, higher low in MACD ────────────────
    anchors = {}
    for i in price_l:
        a = _macd_anchor(macd, macd_l, i, "low", rules["require_macd_pivot"])
        if a:
            anchors[i] = a

    for b in price_l:
        if n - 1 - b > age or b not in anchors:
            continue
        mb_bar, mb_val = anchors[b]
        atr_pct = (atrv[b] / close[b] * 100) if close[b] else 0.0
        min_pct = max(rules["min_price_pct"], rules["atr_frac"] * atr_pct)
        best = None

        for a in price_l:
            if not (mn <= b - a <= mx) or a not in anchors:
                continue
            ma_bar, ma_val = anchors[a]

            # core divergence conditions
            if not (low[b] < low[a] and mb_val > ma_val):
                continue

            # minimum price separation (absolute floor + ATR-relative)
            if (low[a] - low[b]) / low[a] * 100 < min_pct:
                continue

            # minimum MACD separation, as a share of the leg's MACD range
            seg = macd[min(ma_bar, a):max(mb_bar, b) + 1]
            span = float(seg.max() - seg.min())
            if span <= 0:
                continue
            if (mb_val - ma_val) < rules["min_macd_frac"] * span:
                continue

            if rules["zero_line"] and not (ma_val < 0 and mb_val < 0):
                continue

            if rules["require_clean"]:
                # B must be the lowest low of the leg (0.1% undercut tolerance)
                if float(low[a:b + 1].min()) < low[b] * 0.999:
                    continue
                # A must be the MACD low of the leg (2% of span tolerance)
                lo_seg = min(ma_bar, a)
                hi_seg = max(mb_bar, b) + 1
                if float(macd[lo_seg:hi_seg].min()) < ma_val - 0.02 * span:
                    continue

            cand = (a, ma_bar, ma_val, span)
            if best is None or ma_val < best[2]:      # deepest MACD trough wins
                best = cand

        if best:
            a, ma_bar, ma_val, span = best
            out.append(_mk("REGULAR BULLISH", "bullish", df, a, b,
                           ma_bar, ma_val, mb_bar, mb_val, span, cfg))

    # ── Regular bearish: higher high in price, lower high in MACD ─────────────
    anchors = {}
    for i in price_h:
        a = _macd_anchor(macd, macd_h, i, "high", rules["require_macd_pivot"])
        if a:
            anchors[i] = a

    for b in price_h:
        if n - 1 - b > age or b not in anchors:
            continue
        mb_bar, mb_val = anchors[b]
        atr_pct = (atrv[b] / close[b] * 100) if close[b] else 0.0
        min_pct = max(rules["min_price_pct"], rules["atr_frac"] * atr_pct)
        best = None

        for a in price_h:
            if not (mn <= b - a <= mx) or a not in anchors:
                continue
            ma_bar, ma_val = anchors[a]

            if not (high[b] > high[a] and mb_val < ma_val):
                continue
            if (high[b] - high[a]) / high[a] * 100 < min_pct:
                continue

            seg = macd[min(ma_bar, a):max(mb_bar, b) + 1]
            span = float(seg.max() - seg.min())
            if span <= 0:
                continue
            if (ma_val - mb_val) < rules["min_macd_frac"] * span:
                continue

            if rules["zero_line"] and not (ma_val > 0 and mb_val > 0):
                continue

            if rules["require_clean"]:
                if float(high[a:b + 1].max()) > high[b] * 1.001:
                    continue
                lo_seg = min(ma_bar, a)
                hi_seg = max(mb_bar, b) + 1
                if float(macd[lo_seg:hi_seg].max()) > ma_val + 0.02 * span:
                    continue

            if best is None or ma_val > best[2]:      # highest MACD peak wins
                best = (a, ma_bar, ma_val, span)

        if best:
            a, ma_bar, ma_val, span = best
            out.append(_mk("REGULAR BEARISH", "bearish", df, a, b,
                           ma_bar, ma_val, mb_bar, mb_val, span, cfg))

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
def _mk(kind, side, df, a, b, ma_bar, ma_val, mb_bar, mb_val, span, cfg):
    n = len(df)
    sp   = float(df["close"].iloc[b])          # signal price = pivot bar close
    sl   = float(df["low"].iloc[b])
    sh   = float(df["high"].iloc[b])
    atrb = float(df["atr"].iloc[b]) if not pd.isna(df["atr"].iloc[b]) else sp * 0.01

    # Stop sits beyond the swing extreme by the LARGER of 0.3% and 0.3 ATR.
    # A flat percentage buffer is too tight on a high-volatility alt and
    # needlessly wide on a mega-cap; ATR sizes it to the coin.
    buf = max(sp * 0.003, atrb * 0.30)
    if side == "bullish":
        stop = px(sl - buf)
        risk = max(sp - stop, sp * 0.005)
    else:
        stop = px(sh + buf)
        risk = max(stop - sp, sp * 0.005)

    off = pd.Timedelta(hours=DISPLAY_TZ_OFFSET_H)
    def fmt(ts):
        return (ts + off).strftime("%b %d %Y, %I:%M %p")

    # ── Structure metrics, used by the setup score ─────────────────────────────
    if side == "bullish":
        price_gap = (float(df["low"].iloc[a]) - sl)
        zero_ok   = (ma_val < 0 and mb_val < 0)
        macd_gap  = mb_val - ma_val
    else:
        price_gap = (sh - float(df["high"].iloc[a]))
        zero_ok   = (ma_val > 0 and mb_val > 0)
        macd_gap  = ma_val - mb_val

    macd_strength = float(np.clip(macd_gap / span, 0, 1)) if span > 0 else 0.0
    price_atr     = float(price_gap / atrb) if atrb > 0 else 0.0

    # Has momentum actually turned since the pivot? This is the one
    # confirmation kept from v3 — MACD-native, so it needs no second indicator.
    #
    # CAREFUL: the obvious version of this ("is MACD on the correct side of
    # signal at any bar after b") is wrong, and shipped wrong once. If MACD was
    # ALREADY aligned at the pivot it returns true on the very first bar, so a
    # cross that happened well BEFORE the divergence completed gets scored as
    # confirmation of it. A cross means a cross: wrong side at i-1, right side
    # at i. Being already aligned at the pivot is reported separately and
    # scores less, because it is a weaker fact.
    mv, sv = df["macd"].values, df["signal"].values
    aligned_at_pivot = bool(mv[b] > sv[b]) if side == "bullish" else bool(mv[b] < sv[b])
    cross_at = None
    for i in range(b + 1, n):
        crossed = (mv[i] > sv[i] and mv[i - 1] <= sv[i - 1]) if side == "bullish" \
            else (mv[i] < sv[i] and mv[i - 1] >= sv[i - 1])
        if crossed:
            cross_at = fmt(df.index[i])
            break

    return {
        "type": kind, "side": side,
        "implication": "Potential reversal UP" if side == "bullish" else "Potential reversal DOWN",
        "pivot1_time": fmt(df.index[a]), "pivot2_time": fmt(df.index[b]),
        "pivot1_price": px(float(df["low"].iloc[a] if side == "bullish" else df["high"].iloc[a])),
        "pivot2_price": px(sl if side == "bullish" else sh),
        "pivot1_macd": round(ma_val, 8), "pivot2_macd": round(mb_val, 8),
        "signal_price": px(sp),
        "pivot_dist": b - a,
        "bars_ago": n - 1 - b,
        "fresh": (n - 1 - b) <= cfg["fresh"],
        "stop_price": stop,
        "signal_risk": risk,
        "macd_strength": round(macd_strength, 4),
        "price_atr": round(price_atr, 3),
        "zero_ok": bool(zero_ok),
        "macd_cross": cross_at is not None,
        "macd_cross_at": cross_at,
        "macd_aligned_at_pivot": aligned_at_pivot,
        "_b": b,
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


def build_targets(sig, current_price):
    """1R/2R/3R levels.

    Anchored to the signal price while the setup is still near it, and
    re-anchored to spot once price has run — because if you enter now, your
    risk is measured from now, and quoting the original targets would flatter
    the R:R you'd actually get.
    """
    side = sig["side"]
    stop = sig["stop_price"]
    moved = sig.get("status") in ("extended", "ran")
    ref = current_price if moved else sig["signal_price"]

    if side == "bullish":
        risk = max(ref - stop, ref * 0.005)
        tg = {"1r": px(ref + risk), "2r": px(ref + 2 * risk), "3r": px(ref + 3 * risk)}
    else:
        risk = max(stop - ref, ref * 0.005)
        tg = {"1r": px(ref - risk), "2r": px(ref - 2 * risk), "3r": px(ref - 3 * risk)}

    return {"entry_ref_price": px(ref),
            "targets_from": "current" if moved else "signal",
            "risk_pct": round(risk / ref * 100, 2),
            "targets": tg}


# ── Setup score ────────────────────────────────────────────────────────────────
def setup_score(sig, weekly_vol):
    """0–100 description of how textbook the divergence looks.

    READ THIS BEFORE TRUSTING THE NUMBER: it is NOT a win rate and NOT a
    backtested edge. It is a structural description — how large the momentum
    divergence is relative to the leg, how decisive the price break is in ATR
    terms, whether the pivots are sensibly spaced, whether they sit on the
    classic side of the MACD zero line, whether momentum has crossed since,
    and how liquid the pair is. A 90 is a clean textbook picture. It is still
    your call whether to take it.
    """
    s  = round(sig["macd_strength"] * 30)                          # 0–30
    s += round(float(np.clip(sig["price_atr"] / 1.5, 0, 1)) * 15)  # 0–15
    d  = sig["pivot_dist"]                                          # 0–15
    s += 15 if 10 <= d <= 60 else (10 if 6 <= d <= 90 else 5)
    s += 15 if sig["zero_ok"] else 0                                # 0–15
    # a genuine post-pivot cross is worth more than momentum that had already
    # turned before the divergence finished forming
    s += 15 if sig["macd_cross"] else (7 if sig.get("macd_aligned_at_pivot") else 0)

    if weekly_vol:                                                  # 0–10
        s += 10 if weekly_vol >= 700e6 else (7 if weekly_vol >= 250e6 else 4)
    else:
        # Liquidity unknown (the exchange reported no usable volume). Capping
        # at 90 would make those pairs look structurally worse than they are
        # and sort them below equivalent setups, so the structural 0–90 is
        # rescaled onto the same 0–100 axis instead.
        s = round(s * 100 / 90)
    return int(max(0, min(100, s)))


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


# ── Watchlist helpers ──────────────────────────────────────────────────────────
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


def fetch_oi_history(ex, limiter, sym, limit, tries=3):
    """Daily OI history with the same backoff discipline as fetch_ohlcv."""
    for attempt in range(tries):
        try:
            limiter.acquire()
            return ex.fetch_open_interest_history(sym, timeframe="1d", limit=limit)
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


def classify_mover(m):
    """(qualifies, flags, score) for one analysed pair.

    The score ranks how UNUSUAL the day was and how close price sits to a
    decision level. It carries NO direction and is NOT a win rate — the same
    caveat as setup_score, for the same reason: nothing here was backtested.
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


def calendar_time_check(events, reference):
    """Median hour difference between events present in both lists (same
    currency + title, nearest occurrence within a day). None if no overlap.
    Anything but ~0 means JB_TIME_OFFSET is wrong."""
    ref = {}
    for e in reference:
        ref.setdefault((e["currency"], e["title"]), []).append(_cal_parse_ts("iso", e["ts"]))
    deltas = []
    for e in events:
        cands = ref.get((e["currency"], e["title"]))
        if not cands:
            continue
        t = _cal_parse_ts("iso", e["ts"])
        nearest = min(cands, key=lambda c: abs((c - t).total_seconds()))
        d = (t - nearest).total_seconds() / 3600.0
        if abs(d) <= 24:
            deltas.append(d)
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
    by_key = {}
    for i, e in enumerate(jb_events):
        by_key.setdefault((e["currency"], e["title"]), []).append((i, _cal_parse_ts("iso", e["ts"])))
    used, out = set(), []
    for f in ff_events:
        e = dict(f)
        cands = by_key.get((f["currency"], f["title"]))
        if cands:
            t = _cal_parse_ts("iso", f["ts"])
            i, jt = min(cands, key=lambda c: abs((c[1] - t).total_seconds()))
            if i not in used and abs((jt - t).total_seconds()) <= 24 * 3600:
                used.add(i)
                if jb_events[i]["actual"] and not e["actual"]:
                    e["actual"] = jb_events[i]["actual"]
        out.append(e)
    delta = datetime.timedelta(hours=shift or 0.0)
    for i, j in enumerate(jb_events):
        if i not in used:
            out.append(dict(j, ts=_iso_z(_cal_parse_ts("iso", j["ts"]) - delta)))
    out.sort(key=lambda e: (e["ts"], e["currency"], e["title"]))
    return out, shift


def _http_json(url, headers=None, timeout=25):
    h = {"User-Agent": "Mozilla/5.0 (divergence-screener)", "Accept": "application/json"}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_jblanked(key):
    today = datetime.datetime.now(datetime.timezone.utc).date()
    url = (f"{JB_CAL_URL}?from={today - datetime.timedelta(days=1)}"
           f"&to={today + datetime.timedelta(days=CAL_DAYS_AHEAD)}&offset={JB_TIME_OFFSET}")
    try:
        data = _http_json(url, {"Authorization": "Api-Key " + key,
                                "Content-Type": "application/json"})
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError("HTTP 401: key rejected, or the free tier's "
                               "one-call-per-5-minutes limit was hit") from None
        raise
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
    errors, ff_events, jb_events = [], [], []
    # ForexFactory is fetched every time: it is authoritative for this week,
    # the clock reference for JBlanked, and the fallback.
    try:
        ff_events = normalise_events(fetch_ff(), "forexfactory")
    except Exception as e:
        errors.append(f"ForexFactory: {e}")
    if key:
        try:
            jb_events = normalise_events(fetch_jblanked(key), "jblanked")
        except Exception as e:
            errors.append(f"JBlanked: {e}")

    shift = None
    if jb_events and ff_events:
        events, shift = merge_calendars(ff_events, jb_events)
        source = "jblanked+forexfactory"
    elif jb_events:
        events, source = jb_events, "jblanked"
    elif ff_events:
        events, source = ff_events, "forexfactory"
    else:
        events, source = [], None

    meta = {"source": source, "has_key": bool(key),
            "error": "; ".join(errors) if errors else None,
            "time_check_h": shift, "fallback": None}
    if source == "forexfactory":
        meta["fallback"] = ("No JBlanked key set" if not key
                            else "JBlanked failed, using the ForexFactory feed")
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
            data = {"events": list(_cal_state["events"]),
                    "ts": _cal_state["ts"].isoformat() if _cal_state["ts"] else None,
                    "meta": dict(_cal_state["meta"])}
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
    14:05 and one at 15:55 can disagree about the same setup. Pivots need
    pivot_lb bars either side so the open bar can't create one — but it can
    still shift the indicator values everything else is measured against.
    """
    if df is None or df.empty:
        return df
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    if df.index[-1] + pd.Timedelta(minutes=tf_minutes) > pd.Timestamp(now):
        return df.iloc[:-1]
    return df


# ── Dedupe ─────────────────────────────────────────────────────────────────────
def dedupe(sigs):
    """One signal per side per pair.

    Successive pivots of the same side are the same setup seen twice. Keep the
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


# ── State persistence ──────────────────────────────────────────────────────────
def save_state():
    try:
        with _lock:
            w = _state["watch"]
            data = {"results": list(_state["results"]),
                    "ts": _state["ts"].isoformat() if _state["ts"] else None,
                    "meta": dict(_state["meta"]),
                    "watch": {"rows": list(w["rows"]),
                              "ts": w["ts"].isoformat() if w["ts"] else None,
                              "meta": dict(w["meta"])}}
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[state] save failed: {e}")


def load_state():
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        with _lock:
            _state["results"] = data.get("results", [])
            ts = data.get("ts")
            _state["ts"] = datetime.datetime.fromisoformat(ts) if ts else None
            _state["meta"] = data.get("meta", {})
            w = data.get("watch") or {}
            wts = w.get("ts")
            _state["watch"] = {"rows": w.get("rows", []),
                               "ts": datetime.datetime.fromisoformat(wts) if wts else None,
                               "meta": w.get("meta", {})}
        print(f"[state] restored {len(_state['results'])} signal(s) from last scan")
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
        "__OI_LB__":       str(OI_LOOKBACK_D),
        "__OI_MIN__":      format(OI_MIN_CHANGE_PCT, "g"),
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
        return jsonify({"signals": list(_state["results"]),
                        "meta": dict(_state["meta"]),
                        "ts": _state["ts"].isoformat() if _state["ts"] else None})


@app.route("/watch_signals")
def watch_signals_route():
    with _lock:
        w = _state["watch"]
        return jsonify({"rows": list(w["rows"]), "meta": dict(w["meta"]),
                        "ts": w["ts"].isoformat() if w["ts"] else None})


@app.route("/watchlist")
def watchlist_route():
    """OI flow scan — the Sunday routine as one SSE stream.

    Per pair: ~10 daily candles (price baseline + real 7-day volume gate) and
    the daily OI history. Funding for the whole market is one bulk call.
    Roughly half the API weight of a divergence scan.
    """
    exid      = request.args.get("exchange", "okx")
    speed     = request.args.get("speed", "normal")
    min_vol   = float(request.args.get("min_weekly_vol", DEFAULT_MIN_WEEKLY_VOL))
    max_pairs = int(request.args.get("max_pairs", DEFAULT_MAX_PAIRS))
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

        perps = [s for s in mkts
                 if s.endswith("/USDT:USDT") and mkts[s].get("active")
                 and s.split("/")[0] not in STABLE_BASES
                 and not is_tradfi(mkts[s])]

        yield sse("log", {"msg": "Fetching 24h volumes + funding rates (bulk)..."})
        try:
            tickers = main_ex.fetch_tickers()
        except Exception as e:
            yield sse("error", {"msg": f"volume fetch failed: {e}"}); return
        funding = fetch_funding_map(main_ex)
        if not funding:
            yield sse("log", {"msg": "No bulk funding data — health flags will be blank"})

        vol24 = {}
        for s in perps:
            v = usd_volume(tickers.get(s))
            if v:
                vol24[s] = v
        pre_floor = (min_vol / 7.0) * PREGATE_SLACK
        pairs = sorted([s for s in vol24 if vol24[s] >= pre_floor],
                       key=lambda s: vol24[s], reverse=True)[:max_pairs]
        yield sse("log", {"msg": f"{len(pairs)} candidates - measuring {OI_LOOKBACK_D}-day OI change"})

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
        counter = {"n": 0, "analysed": 0, "vol_reject": 0, "no_oi": 0,
                   "quiet": 0, "errors": 0}
        clock = threading.Lock()

        def scan_pair(sym):
            base = sym.split("/")[0]
            try:
                ex = worker_ex()
                raw = _to_df(fetch_ohlcv(ex, limiter, sym, "1d",
                                         limit=OI_LOOKBACK_D + 5))
                if raw is None or raw.empty:
                    with clock:
                        counter["no_oi"] += 1
                    return
                live_price = float(raw["close"].iloc[-1])
                df = drop_forming_candle(raw, 1440)
                if df is None or len(df) < 2:
                    with clock:
                        counter["no_oi"] += 1
                    return

                wk_vol = weekly_usd_volume(df, vol24.get(sym), 1440)
                if wk_vol is None or wk_vol < min_vol:
                    with clock:
                        counter["vol_reject"] += 1
                    return

                hist = fetch_oi_history(ex, limiter, sym, OI_LOOKBACK_D + 4)
                hist = [p for p in (hist or []) if p.get("timestamp")]
                if len(hist) < OI_MIN_HISTORY:
                    # fresh listing or venue gap — OI ramping from zero is the
                    # method's classic false positive, so it is skipped loudly
                    with clock:
                        counter["no_oi"] += 1
                    return
                hist.sort(key=lambda p: p["timestamp"])

                # Baseline = the OI point closest to (now − lookback). Daily
                # points can be sparse on some venues, so "closest" with a
                # ±36h tolerance beats indexing back a fixed count.
                now_ms = hist[-1]["timestamp"]
                target = now_ms - OI_LOOKBACK_D * 86_400_000
                past = min(hist[:-1], key=lambda p: abs(p["timestamp"] - target))
                if abs(past["timestamp"] - target) > 36 * 3_600_000:
                    with clock:
                        counter["no_oi"] += 1
                    return

                # price for amount→USD conversion at the baseline: the daily
                # close of that same day, falling back to live price
                day_close = {int(ts.timestamp() * 1000) // 86_400_000: float(c)
                             for ts, c in df["close"].items()}
                past_px = day_close.get(past["timestamp"] // 86_400_000, live_price)
                oi_now  = oi_point_usd(hist[-1], live_price)
                oi_then = oi_point_usd(past, past_px)
                if not oi_now or not oi_then:
                    with clock:
                        counter["no_oi"] += 1
                    return

                oi_chg = (oi_now - oi_then) / oi_then * 100.0
                span = min(OI_LOOKBACK_D + 1, len(df))
                price_chg = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-span]) - 1) * 100.0

                with clock:
                    counter["analysed"] += 1
                if abs(oi_chg) < OI_MIN_CHANGE_PCT:
                    with clock:
                        counter["quiet"] += 1
                    return

                fund_day = funding.get(sym)
                quad, flags = classify_flow(price_chg, oi_chg, fund_day)
                avg_daily_vol = wk_vol / 7.0
                oi_vol_ratio = oi_now / avg_daily_vol if avg_daily_vol > 0 else None
                if oi_vol_ratio and oi_vol_ratio >= OI_VOL_RATIO_HIGH:
                    flags.append("HIGH OI/VOL")

                q.put(("result", {
                    "pair": sym, "base": base,
                    "oi_usd": round(oi_now, 0),
                    "oi_chg_pct": round(oi_chg, 1),
                    "price_chg_pct": round(price_chg, 1),
                    "quadrant": quad, "flags": flags,
                    "funding_day_pct": round(fund_day, 4) if fund_day is not None else None,
                    "oi_vol_ratio": round(oi_vol_ratio, 2) if oi_vol_ratio else None,
                    "weekly_volume": round(wk_vol, 2),
                    "current_price": px(live_price),
                }))
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
            meta = {"exchange": exid, "lookback_d": OI_LOOKBACK_D,
                    "min_weekly_vol": min_vol, "dispatched": len(pairs),
                    "analysed": c["analysed"], "vol_rejected": c["vol_reject"],
                    "no_oi": c["no_oi"], "quiet": c["quiet"],
                    "errors": c["errors"],
                    "complete": done.is_set() and q.empty()}
            with _lock:
                _state["watch"] = {"rows": rows,
                                   "ts": datetime.datetime.now(datetime.timezone.utc),
                                   "meta": meta}
            save_state()
            return meta

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
    for sym in mkts:
        if not (sym.endswith("/USDT:USDT") and mkts[sym].get("active")):
            continue
        base = sym.split("/")[0]
        if base in STABLE_BASES or is_tradfi(mkts[sym]):
            continue
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
    tf         = request.args.get("tf", "4h")
    strictness = request.args.get("strictness", "balanced")
    speed      = request.args.get("speed", "normal")
    min_vol    = float(request.args.get("min_weekly_vol", DEFAULT_MIN_WEEKLY_VOL))
    max_pairs  = int(request.args.get("max_pairs", DEFAULT_MAX_PAIRS))
    only_syms  = [s.strip().upper() for s in request.args.get("symbols", "").split(",") if s.strip()]

    if tf not in TF_CFG:
        tf = "4h"
    if strictness not in STRICTNESS:
        strictness = "balanced"
    if speed not in SPEED:
        speed = "normal"

    cfg   = TF_CFG[tf]
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

        perps = [s for s in mkts
                 if s.endswith("/USDT:USDT") and mkts[s].get("active")
                 and s.split("/")[0] not in STABLE_BASES
                 and not is_tradfi(mkts[s])]
        yield sse("log", {"msg": f"{len(perps)} active linear USDT perps on {exid.upper()}"})

        yield sse("log", {"msg": "Fetching 24h volumes (one bulk call)..."})
        try:
            tickers = main_ex.fetch_tickers()
        except Exception as e:
            yield sse("error", {"msg": f"volume fetch failed: {e}"}); return

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
        workers = spd["workers"]
        yield sse("log", {"msg": f"Scanning {len(pairs)} pairs on {cfg['label']} · "
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
        counter = {"n": 0, "analysed": 0, "vol_reject": 0, "no_data": 0, "errors": 0}
        clock = threading.Lock()

        def scan_pair(sym):
            base = sym.split("/")[0]
            try:
                ex = worker_ex()
                raw = fetch_df_deep(ex, limiter, sym, tf, cfg["bars"])
                if raw is None or len(raw) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return
                live_price = float(raw["close"].iloc[-1])

                # Drop the open candle BEFORE measuring volume. Getting this
                # order wrong is not cosmetic: weekly_usd_volume divides the
                # 7-day notional by the last 24h of notional, and on the 1D
                # timeframe "the last 24h" is a single bar. If that bar is the
                # one still forming, the denominator is a fraction of a day's
                # volume and the ratio blows up — a coin scanned at 02:00 UTC
                # reported ~10x its real weekly volume and sailed through the
                # liquidity gate it should have failed.
                df = drop_forming_candle(raw, cfg["minutes"])
                if df is None or len(df) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return

                wk_vol = weekly_usd_volume(df, vol24.get(sym), cfg["minutes"])
                if not only_syms and (wk_vol is None or wk_vol < min_vol):
                    with clock:
                        counter["vol_reject"] += 1
                    return

                df = pd.concat([df, compute_indicators(df)], axis=1).dropna()
                if len(df) < 80:
                    with clock:
                        counter["no_data"] += 1
                    return

                with clock:
                    counter["analysed"] += 1

                found = []
                for sig in detect_divergences(df, cfg, rules):
                    sig.update(evaluate_status(df, sig, live_price))
                    found.append(sig)

                for sig in dedupe(found):
                    sig.pop("_b", None)
                    sig.update(build_targets(sig, live_price))
                    sig.update({
                        "pair": sym, "base": base, "tf": tf,
                        "current_price": px(live_price),
                        "weekly_volume": round(wk_vol, 2) if wk_vol else None,
                        "score": setup_score(sig, wk_vol),
                    })
                    q.put(("result", sig))
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
            # reports what reached detect_divergences, and every other bucket
            # is reported alongside it instead of disappearing.
            with clock:
                c = dict(counter)
            meta = {"exchange": exid, "tf": tf, "strictness": strictness,
                    "min_weekly_vol": min_vol, "universe": len(perps),
                    "dispatched": len(pairs), "scanned": c["analysed"],
                    "vol_rejected": c["vol_reject"], "no_data": c["no_data"],
                    "errors": c["errors"], "no_ticker": no_ticker,
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
        pre_floor = (min_vol / 7.0) * PREGATE_SLACK
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
                   "quiet": 0, "errors": 0}
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
                if wk_vol is None or wk_vol < min_vol:
                    with clock:
                        counter["vol_reject"] += 1
                    return

                m = analyse_mover(df, live_price)
                if m is None:
                    with clock:
                        counter["no_data"] += 1
                    return

                with clock:
                    counter["analysed"] += 1

                qualifies, flags, score = classify_mover(m)
                if not qualifies:
                    with clock:
                        counter["quiet"] += 1
                    return

                m.update({
                    "pair": sym, "base": base,
                    "direction": "UP" if m["chg_1d_pct"] >= 0 else "DOWN",
                    "flags": flags, "score": score,
                    "weekly_volume": round(wk_vol, 2),
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
.shell{display:grid;grid-template-rows:48px 1fr;height:100vh;overflow:hidden}
.topbar{display:flex;align-items:center;background:var(--sf);border-bottom:1px solid var(--bd);z-index:20}
.t-logo{font-family:var(--mono);font-size:12px;font-weight:600;letter-spacing:.12em;color:var(--acc);padding:0 18px;flex-shrink:0}
.t-div{width:1px;height:16px;background:var(--bd2)}
.nav{display:flex;height:48px;padding:0 8px}
.nav-btn{height:100%;padding:0 18px;font-size:12px;font-weight:500;color:var(--tx3);cursor:pointer;border:none;border-bottom:2px solid transparent;background:none;transition:color .15s,border-color .15s;letter-spacing:.04em}
.nav-btn:hover{color:var(--tx2)}
.nav-btn.active{color:var(--tx);border-bottom-color:var(--acc)}
.t-right{margin-left:auto;display:flex;align-items:center;gap:8px;padding-right:18px}
.sdot{width:6px;height:6px;border-radius:50%;background:var(--tx3);transition:background .4s,box-shadow .4s}
.sdot.live{background:var(--bull);box-shadow:0 0 8px var(--acc);animation:blink 2s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
.slbl{font-family:var(--mono);font-size:11px;color:var(--tx3);transition:opacity .2s}
.slbl.fd{opacity:.3}
.page{display:none}
.page.active{display:flex;flex-direction:column}
#pg-dash.active,#pg-signals.active,#pg-watch.active{display:block;overflow-y:auto;padding:24px}
#pg-scan.active{display:grid;grid-template-columns:286px 1fr;overflow:hidden}
#pg-dash,#pg-signals,#pg-watch{scrollbar-width:thin;scrollbar-color:var(--bd2) transparent}
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
tr.dim td{opacity:.45}
tr:hover td{background:var(--sf2)}
@keyframes ri{from{opacity:0;transform:translateX(-5px)}to{opacity:1;transform:translateX(0)}}
tbody tr{animation:ri .3s var(--eout) both}
.pb{display:flex;flex-direction:column;gap:2px}
.pb-pair{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--tx)}
.pb-q{font-family:var(--mono);font-size:9px;color:var(--tx3)}
.bdg{display:inline-flex;align-items:center;padding:3px 7px;border-radius:4px;font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.04em;white-space:nowrap}
.bb{background:var(--bull-bg);color:var(--bull);border:1px solid #1a4020}
.br{background:var(--bear-bg);color:var(--bear);border:1px solid #4a1a22}
.bx{background:var(--fresh-bg);color:var(--fresh);border:1px solid #4a3500}
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
.status{display:inline-flex;flex-direction:column;gap:1px;padding:5px 9px;border-radius:5px;font-family:var(--mono);min-width:118px;max-width:100%;overflow:hidden}
.tlbl{font-size:11px;font-weight:700;white-space:nowrap}
.tsub{font-size:9px;font-weight:400;opacity:.8;margin-top:1px;white-space:normal;line-height:1.3}
.st-active{background:#0D2E1F;color:#34D399;border:1px solid #1a4020}
.st-extended{background:#2D1500;color:#FB923C;border:1px solid #4a2800}
.st-ran{background:#2D1F06;color:#FBBF24;border:1px solid #4a3500}
.st-invalidated{background:#1A2438;color:#4F6380;border:1px solid var(--bd2)}
.volc{font-family:var(--mono);font-size:11px;color:var(--tx2)}
.volc small{display:block;font-size:9px;color:var(--tx3)}
.tgt-row{font-family:var(--mono);font-size:10px;display:flex;gap:6px;flex-wrap:wrap}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:10px;color:var(--tx3);font-family:var(--mono);font-size:12px}
.empty-icon{font-size:28px;opacity:.3}
.greet{display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center;padding:48px 40px}
.g-hl{font-family:var(--sans);font-size:42px;font-weight:600;color:var(--tx);letter-spacing:-.02em;line-height:1.15}
.g-sub{font-family:var(--sans);font-size:16px;color:var(--tx3);max-width:380px;line-height:1.7}
.g-cta{margin-top:16px;padding:14px 52px;background:var(--acc);color:#000;border:none;border-radius:7px;font-family:var(--sans);font-size:14px;font-weight:600;cursor:pointer;transition:opacity .2s,transform .15s,box-shadow .2s}
.g-cta:hover{opacity:.88;box-shadow:0 0 28px rgba(240,96,138,.4)}
.g-cta:active{transform:scale(.97)}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
.kpi{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:14px 16px}
.kpi-lbl{font-size:10px;color:var(--tx3);margin-bottom:4px;letter-spacing:.06em;text-transform:uppercase}
.kpi-val{font-family:var(--mono);font-size:28px;font-weight:600;color:var(--tx)}
.kpi-sub{font-size:11px;color:var(--tx3);margin-top:2px}
.dash-sec{margin-bottom:28px}
.dash-sec-ttl{font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--tx3);margin-bottom:12px}
.ready-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.sig-card{background:var(--sf);border:1px solid var(--bd);border-radius:10px;overflow:hidden}
.sc-timing{padding:8px 14px;font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.04em}
.sc-body{padding:14px}
.sc-header{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap}
.sc-pair{font-family:var(--mono);font-size:16px;font-weight:600;color:var(--tx)}
.sc-meta{display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px}
.sml{color:var(--tx3);font-size:10px}.smv{color:var(--tx2);font-family:var(--mono)}
.smv.bull{color:var(--bull)}.smv.bear{color:var(--bear)}
.qbar-wrap{margin-top:10px;padding-top:10px;border-top:1px solid var(--bd)}
.qbar-lbl{display:flex;justify-content:space-between;font-size:10px;color:var(--tx3);margin-bottom:4px}
.qbar-bg{height:4px;background:var(--sf3);border-radius:2px;overflow:hidden}
.qbar-fill{height:100%;border-radius:2px}
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
.sig-list{display:flex;flex-direction:column;gap:8px}
.srow{display:grid;grid-template-columns:132px 165px 120px 60px 96px 116px 1fr;align-items:center;gap:10px;background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:12px 14px;transition:border-color .15s}
.srow:hover{border-color:var(--bd2)}
.srow.dim{opacity:.5}
.srow-pair{display:flex;flex-direction:column;gap:3px}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--bd2);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--tx3)}
#pg-movers.active{display:block;overflow-y:auto;padding:24px}
#pg-movers{scrollbar-width:thin;scrollbar-color:var(--bd2) transparent}

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
.mrank{display:inline-flex;align-items:center;justify-content:center;width:21px;height:21px;border-radius:5px;background:var(--sf3);border:1px solid var(--bd2);font-family:var(--mono);font-size:10px;font-weight:700;color:var(--tx3)}
.mrank.top{background:var(--acc);border-color:var(--acc);color:#000}
.mf-vol{background:#2D1F06;color:#FBBF24;border:1px solid #4a3500}
.mf-rng{background:#1B1030;color:#C4A0FF;border:1px solid #3A2A5A}
.mf-hi{background:var(--bull-bg);color:var(--bull);border:1px solid #1a4020}
.mf-lo{background:var(--bear-bg);color:var(--bear);border:1px solid #4a1a22}
.mf-move{background:#0F2A3A;color:#7DD3FC;border:1px solid #1F4A66}
.mpos{position:relative;width:88px;height:5px;background:var(--sf3);border-radius:3px}
.mpos i{position:absolute;top:-3px;width:3px;height:11px;border-radius:2px;background:var(--tx);transform:translateX(-1px)}
.mlv{font-family:var(--mono);font-size:10px;color:var(--tx2);white-space:nowrap;line-height:1.6}
.mlv small{color:var(--tx3);font-size:9px}
</style>
</head>
<body>
<div class="shell">

<header class="topbar">
  <span class="t-logo">MACD&middot;DIV</span>
  <div class="t-div"></div>
  <nav class="nav">
    <button class="nav-btn active" data-pg="dash">Dashboard</button>
    <button class="nav-btn" data-pg="scan">Scanner</button>
    <button class="nav-btn" data-pg="signals">Signals</button>
    <button class="nav-btn" data-pg="watch">Watchlist</button>
    <button class="nav-btn" data-pg="movers">Movers</button>
    <button class="nav-btn" data-pg="calendar">Calendar</button>
    <button class="nav-btn" data-pg="routine">Routine</button>
    <button class="nav-btn" data-pg="learn">Method</button>
  </nav>
  <div class="t-right">
    <div class="sdot" id="sdot"></div>
    <span class="slbl" id="slbl">Ready</span>
  </div>
</header>

<div id="pg-dash" class="page active">
  <div id="dash-loader" style="display:flex;align-items:center;justify-content:center;height:180px;color:var(--tx3);font-family:var(--mono);font-size:12px">Loading last scan...</div>
  <div id="dash-content" style="display:none">
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-lbl">Live setups</div><div class="kpi-val" id="kpi-live">-</div><div class="kpi-sub" id="kpi-live-sub">Stop not yet taken out</div></div>
      <div class="kpi"><div class="kpi-lbl">Regular bullish</div><div class="kpi-val" id="kpi-bull" style="color:var(--bull)">-</div><div class="kpi-sub">Reversal up</div></div>
      <div class="kpi"><div class="kpi-lbl">Regular bearish</div><div class="kpi-val" id="kpi-bear" style="color:var(--bear)">-</div><div class="kpi-sub">Reversal down</div></div>
      <div class="kpi"><div class="kpi-lbl">Pairs analysed</div><div class="kpi-val" id="kpi-scan">-</div><div class="kpi-sub" id="kpi-scan-sub">No scan yet</div></div>
    </div>
    <div id="scan-audit" class="audit"></div>
    <div class="cal-strip" id="dash-cal" style="display:none"></div>
    <div class="dash-sec">
      <div class="dash-sec-ttl">Freshest setups &mdash; newest pivots, stop intact</div>
      <div class="ready-grid" id="ready-grid"></div>
    </div>
    <div class="dash-sec">
      <div class="dash-sec-ttl">All live signals from the last scan (sorted by setup score)</div>
      <div class="sig-list" id="dash-sig-list"></div>
    </div>
  </div>
</div>

<div id="pg-scan" class="page">
  <aside class="sb">
    <div class="ss"><div class="fl">Exchange</div>
      <select id="exchange"><option value="okx">OKX</option><option value="binance">Binance</option><option value="bybit">Bybit</option></select></div>

    <div class="ss"><div class="fl">Timeframe</div>
      <div class="trow" id="tfG">
        <button class="tbtn active" data-val="4h">4H</button>
        <button class="tbtn" data-val="1d">1D</button>
      </div>
      <div class="hint">1D pivots are wider and cleaner &mdash; usually the better fit for a week-long hold.</div></div>

    <div class="ss"><div class="fl">Minimum weekly volume ($M)</div>
      <div class="nw"><button class="nb" onclick="stepVol(-10)">-</button><input type="number" id="minVol" value="__DEFAULT_VOL_M__" min="5" max="2000" step="10"><button class="nb" onclick="stepVol(10)">+</button></div>
      <div class="hint">Rolling 7-day traded value. $70M/week is about $10M/day.</div>
      <button class="gbtn" id="uniBtn" style="margin-top:8px" onclick="checkUniverse()">How many coins pass?</button>
      <div id="uniOut" class="hint"></div></div>

    <div class="ss"><div class="fl">Detection strictness</div>
      <div class="trow" id="strG">
        <button class="tbtn" data-val="loose">Loose</button>
        <button class="tbtn active" data-val="balanced">Balanced</button>
        <button class="tbtn" data-val="strict">Strict</button>
      </div>
      <div class="hint" id="strHint">MACD pivot must align with the price pivot, the divergence line must be unbroken, and the move must clear a minimum size.</div></div>

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
    <div class="rbar">
      <span class="rtitle">Signals</span>
      <span class="spill sbull" id="stBull">&#9650; <span class="num" id="cBull">0</span> bullish</span>
      <span class="spill sbear" id="stBear">&#9660; <span class="num" id="cBear">0</span> bearish</span>
      <span class="spill sfresh" id="stFresh">&#9889; <span class="num" id="cFresh">0</span> fresh</span>
      <div class="fbar">
        <button class="ftab active" data-f="live" onclick="setF(this)">Live</button>
        <button class="ftab" data-f="fresh" onclick="setF(this)">Fresh</button>
        <button class="ftab" data-f="bull" onclick="setF(this)">Bullish</button>
        <button class="ftab" data-f="bear" onclick="setF(this)">Bearish</button>
        <button class="ftab" data-f="all" onclick="setF(this)">All</button>
      </div>
    </div>
    <div class="tw">
      <div class="empty" id="emS">
        <span class="empty-icon">&#9672;</span><span>Run a scan to find divergences</span>
      </div>
      <table id="resT" style="display:none">
        <thead><tr>
          <th>Pair</th>
          <th title="Where price is now relative to the signal. At the signal = still near the pivot. Extended = moved on without the stop or 1R being hit. Ran = reached 1R or better. Invalidated = traded through the stop.">Status</th>
          <th title="Regular bullish: price makes a lower low while MACD makes a higher low. Regular bearish: price makes a higher high while MACD makes a lower high. Both are counter-trend reversal signatures.">Type</th>
          <th class="sortable sorted" data-sort="score" title="How textbook the structure is, 0-100: divergence size vs the leg (30) + decisiveness of the price break in ATR terms (15) + pivot spacing (15) + classic side of the MACD zero line (15) + MACD crossing its signal line since the pivot (15) + liquidity (10). Describes the picture. Not a win rate, not backtested.">Setup score &#8595;</th>
          <th class="sortable" data-sort="vol" title="Rolling 7-day traded value in USD, measured from the candles. The liquidity gate.">Weekly vol</th>
          <th class="sortable" data-sort="age" title="Time of pivot B, the bar that completed the divergence. FRESH = it completed within the last few bars of this timeframe.">Signal date</th>
          <th title="Just past pivot B (its low for bullish, its high for bearish) plus an ATR-sized buffer. Price trading through it means the divergence has failed. Risk = stop distance as a % of the entry reference.">Stop</th>
          <th title="Multiples of the stop distance. Anchored to the signal price while the setup is still near it; re-anchored to the live price once it has run, because if you enter now your risk is measured from now.">Targets (1R / 2R / 3R)</th>
        </tr></thead>
        <tbody id="resB"></tbody>
      </table>
    </div>
  </main>
</div>

<div id="pg-signals" class="page">
  <div class="sig-hdr">
    <div><div class="sig-ttl">Signals</div><div class="sig-sub" id="sig-sub">Run a scan first</div></div>
  </div>
  <div class="sig-filters">
    <button class="sig-ftab active" data-sf="live" onclick="setSF(this)">Live</button>
    <button class="sig-ftab" data-sf="fresh" onclick="setSF(this)">Fresh</button>
    <button class="sig-ftab" data-sf="active" onclick="setSF(this)">At the signal</button>
    <button class="sig-ftab" data-sf="bull" onclick="setSF(this)">Bullish</button>
    <button class="sig-ftab" data-sf="bear" onclick="setSF(this)">Bearish</button>
    <button class="sig-ftab" data-sf="all" onclick="setSF(this)">All</button>
  </div>
  <div class="sig-list" id="sigList"><div class="no-sig">No signals yet - run a scan.</div></div>
</div>

<div id="pg-watch" class="page">
  <div class="sig-hdr">
    <div><div class="sig-ttl">Watchlist &mdash; OI flow</div>
    <div class="sig-sub" id="watch-sub">Where did open interest move over the last 7 days? Uses the Scanner's volume floor and speed.</div></div>
  </div>
  <div class="wbar">
    <select id="wExchange"><option value="okx">OKX</option><option value="binance">Binance</option><option value="bybit">Bybit</option></select>
    <button class="rbtn" id="wRunBtn" onclick="startWatch()">Scan OI Flow</button>
    <span class="wprog" id="wProg"></span>
  </div>
  <div class="sig-filters">
    <button class="sig-ftab active" data-wf="all" onclick="setWF(this)">All movers</button>
    <button class="sig-ftab" data-wf="longs" onclick="setWF(this)">New longs</button>
    <button class="sig-ftab" data-wf="shorts" onclick="setWF(this)">New shorts</button>
    <button class="sig-ftab" data-wf="squeeze" onclick="setWF(this)">Squeeze fuel</button>
    <button class="sig-ftab" data-wf="spot" onclick="setWF(this)">Spot-led</button>
  </div>
  <div id="watch-audit" class="audit" style="margin:0 0 12px"></div>
  <div class="tw" style="overflow:visible">
    <table id="wT" style="display:none">
      <thead><tr>
        <th>Coin</th>
        <th title="Open interest now vs about __OI_LB__ days ago, in USD. A pair is listed only if this moved at least __OI_MIN__% either way. OI only rises when new positions are opened, which is the whole reason to look.">&Delta;OI 7d</th>
        <th title="Current open interest in USD. Underneath: OI as a multiple of average daily volume. Above __OI_VOL__x, positions cannot exit without moving the price.">OI now</th>
        <th title="Close-to-close price change over the same __OI_LB__-day window.">&Delta;Price 7d</th>
        <th title="Sign of price and OI together. Price up + OI up = NEW LONGS. Price down + OI up = NEW SHORTS. OI falling means positions closing (SHORT COVERING or CAPITULATION), shown dimmed: movement without new conviction. Hover a badge for detail.">Quadrant</th>
        <th title="Funding rate normalised to % per day (intervals differ per contract). Neutral is about 0.03%/day. Positive = longs pay shorts.">Funding /day</th>
        <th title="Health read from funding and crowding. Hover a tag for its definition and threshold.">Read</th>
        <th title="Rolling 7-day traded value in USD. The liquidity gate.">Weekly vol</th>
      </tr></thead>
      <tbody id="wB"></tbody>
    </table>
    <div class="no-sig" id="wEmpty">No OI scan yet &mdash; hit Scan OI Flow. Sunday is the classic day to look.</div>
  </div>
</div>

<div id="pg-movers" class="page">
  <div class="sig-hdr">
    <div><div class="sig-ttl">Movers &mdash; daily watchlist</div>
    <div class="sig-sub" id="mv-sub">Unusual volume, unusual range, or a big day &mdash; measured on completed daily bars. Crypto perps only. Uses the Scanner's volume floor, speed and max-pairs.</div></div>
  </div>
  <div class="wbar">
    <select id="mvExchange"><option value="okx">OKX</option><option value="binance">Binance</option><option value="bybit">Bybit</option></select>
    <button class="rbtn" id="mvRunBtn" onclick="startMovers()">Scan Movers</button>
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
  </div>
  <div id="cal-audit" class="audit" style="margin:0 0 12px"></div>
  <div class="cal-grid" id="cal-grid"></div>
  <div class="dash-sec">
    <div class="dash-sec-ttl" id="cal-day-ttl">Select a day</div>
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

<div id="pg-routine" class="page" style="overflow-y:auto;padding:24px">
  <div class="dash-sec">
    <div class="dash-sec-ttl">Daily routine &mdash; how to use the three scans together</div>
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75;margin-bottom:18px">
      Movers finds what moved, the Scanner finds where momentum disagrees with price, and the Watchlist
      says whether new money is behind it. Run them in that order, once a day, after the daily candle has
      closed. About fifteen minutes.
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Before you run anything</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Run after the daily close.</b> Perp daily candles close at 00:00 UTC, which is 08:00 Manila. Every scan reads the last <b>completed</b> daily bar, so a run at 07:00 Manila still describes the day before yesterday. Aim for 08:00 to 10:00 Manila.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Pick the exchange with the deepest book.</b> Binance carries the most volume on most pairs. Some majors trade thinly on other venues and get volume-rejected there for no chart reason. Each tab has its own exchange dropdown; keep them the same.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>One volume floor for everything.</b> The Min 7-day volume box on the Scanner tab applies to all three scans. Default is $70M a week, about $10M a day. Names under that do not exist to the screener. If a coin you expected is missing, check the audit line under the table before assuming there was no setup.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Read the audit line every time.</b> It says how many pairs were scanned, volume-rejected, errored, and whether the run finished. A short list from an unfinished scan is not a quiet market.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Check the Calendar tab.</b> A fresh entry sized into a CPI or FOMC print is a coin flip, not a setup. If the next market mover is inside 24 hours, wait for it or size for it. The Dashboard strip shows the countdown.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Step 1 &mdash; Movers: what did something unusual yesterday</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Scan Movers, keep the All filter, read by score.</b> Top 4 is a shortcut, not the list. The score ranks how unusual the day was and how close price sits to a 7-day boundary. It has no direction.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Extended runs are the rows flagged ABOVE 7D HIGH or AT RANGE HIGH with a big 1-day change.</b> The At range high filter isolates them. These names have already run for days. Treat them as fade-or-wait candidates, not chases, until the Scanner shows a bearish divergence or price gives back the prior-day low.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Breakdowns are the mirror.</b> BELOW 7D LOW or AT RANGE LOW with a big negative day. Same logic, other side.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Levels come with the row.</b> PDH and PDL are yesterday's high and low, the 7D range is the box price is in, and the position bar shows where in that box it sits now. Write the level down; that is the part a watchlist poster leaves out.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Known blind spot.</b> A coin that had a quiet day is dropped even if it is sitting in a clean consolidation after a breakout. Movers will not show you the pullback. Step 2 does.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Step 2 &mdash; Scanner on 4H: where momentum disagrees with price</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Run 4H on Balanced.</b> Loose finds more and most of it is junk. Strict adds the zero-line rule and misses real setups. Balanced is the default for a reason.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>On the Signals tab, look at Fresh first, then At the signal.</b> Fresh means the second pivot formed within the last 12 bars, two days on 4H. At the signal means price is still within about 2% of where the divergence completed. Those two filters are your entry candidates.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Ran and invalidated are history.</b> Ran means the move already happened and the targets shown are re-anchored to spot. Invalidated means price traded through the stop. Neither is an entry. They are useful only for checking what the scanner saw before a coin appeared on someone's list.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Bullish rows are your pullback candidates.</b> A regular bullish divergence on 4H during a dip is the structural version of a first pullback after the impulse. This is where the pullback names show up, usually a few days before they get posted.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Bearish rows on extended names are your fade candidates.</b> Cross-check against Step 1: a name at its range high with a fresh bearish divergence is the only version of shorting the run that this screener will vouch for.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>Then 1D for context.</b> Same reading, bigger picture. Fresh on 1D means within 3 bars. A 4H setup that agrees with a 1D one is worth more than either alone.</div></div>
      <div class="rule"><div class="rule-n">07</div><div class="rule-b"><b>Symbols override for a specific name.</b> Type tickers into the Symbols box to scan only those. The volume floor is skipped for an override, so a thin name you are curious about still gets analysed.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Step 3 &mdash; Watchlist (OI flow): is new money behind it</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Qualify names here, do not hunt for them.</b> This tab is ranked by 7-day open-interest change and reads best on a Sunday. Use it to check the names you already have from Steps 1 and 2.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>For an extended run you want NEW LONGS with SPOT-LED.</b> Price up, OI up, funding near zero: the run is being bought in spot while shorts fight it. CROWDED LONGS is the same picture with latecomer leverage paying rich funding. A fade gets stronger with CROWDED, weaker with SPOT-LED.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>For a breakdown, NEW SHORTS with SQUEEZE FUEL is the warning.</b> Shorts crowded and paying deeply negative funding can be squeezed. A pullback long into that is a better trade than a short.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>HIGH OI/VOL means nobody can leave quietly.</b> Expect violent moves either way. Size down.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Step 4 &mdash; Build the shortlist</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Three to five names, no more.</b> Split them into two buckets. <b>Extended</b>: wait for a fade signal or a reclaim. <b>Pullback</b>: a fresh bullish divergence or a hold of the prior-day low.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Each name gets a level, a stop and a target before the session starts.</b> From Movers: PDH, PDL, the 7D range. From the Scanner: the stop price and the 1R, 2R, 3R levels. If you cannot fill those in, the name is not a setup, it is a ticker.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Scores rank, they do not predict.</b> Both scores describe structure. Neither is a backtested win rate. A 90 is a clean picture, not a promise.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Step 5 &mdash; Compare with the mentor's poster</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Two minutes a day.</b> For each name on the poster, note which tab surfaced it, and if none did, why: volume-rejected, quiet day, or a real miss. The audit line answers the first one.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Volume-rejected is not a miss.</b> A name that trades under the floor is excluded by design. Decide once whether you want thin names, and if so lower the floor with your eyes open: thin book, wide spreads, funding-driven drift.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>A real miss is a name with a clear structure that no scan flagged.</b> Those are the ones to write down. If the same kind of miss repeats, the screener needs a new flag, not a lower floor.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">If something looks wrong</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Partial scan warning.</b> The tab was closed mid-run or the exchange rate-limited it. Re-run at the Safe speed before trusting the table.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>A ticker says NOT LISTED.</b> The exchange you chose does not carry that perp. Try another exchange.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Nothing fresh anywhere.</b> Normal in a one-way market: regular divergences need a trend extreme to argue with. Movers still works on those days.</div></div>
    </div>
  </div>
</div>

<div id="pg-learn" class="page" style="overflow-y:auto;padding:24px">
  <div class="dash-sec">
    <div class="dash-sec-ttl">What this screener looks for</div>
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75;margin-bottom:18px">
      Regular divergence only &mdash; price and momentum disagreeing at a trend extreme, which is the
      classic <b style="color:var(--tx)">reversal</b> signature. Hidden (continuation) divergences are
      not scanned. Between pivot A and pivot B the price line and the MACD line move in opposite
      directions; that mismatch is the whole signal.
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
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Completed candles only.</b> The still-forming bar is dropped before anything is calculated, so a scan at 14:05 and one at 15:55 agree about the same setup. Nothing repaints.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Real swing extremes.</b> Troughs come from the candle low and peaks from the high &mdash; not the close. Divergence lines get drawn from wicks, so pivots found on closes sit on the wrong bar.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>MACD has to be pivoting too.</b> A price pivot only counts if a MACD pivot sits within 3 bars of it, and the MACD value used is that pivot's own extreme. Momentum often turns a bar or two off price &mdash; demanding the same bar loses real setups, demanding nothing accepts bars where MACD was not swinging at all.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>The line must be clean.</b> No bar between A and B may undercut B's low (bullish) or exceed B's high (bearish), and no bar's MACD may break past A's extreme. If either happens, the swing you would have drawn is not actually there.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Minimum size.</b> The price difference must clear both a fixed floor and a fraction of ATR, and the MACD difference must be a meaningful share of the MACD range across the leg. This is what kills micro-divergences that look like nothing on the chart.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>Every prior pivot is tested, not just the nearest.</b> The nearest prior swing is usually a shallow intermediate dip that does not diverge. Pivot B is compared against every pivot in the lookback window, and among those that pass, the deepest MACD extreme wins &mdash; the line you would draw by hand.</div></div>
      <div class="rule"><div class="rule-n">07</div><div class="rule-b"><b>Liquidity gate.</b> Rolling 7-day traded value must clear the floor you set. Measured from the candles as a ratio against the exchange's own 24h figure, so it does not depend on guessing whether a venue reports volume in coins, contracts or dollars.</div></div>
    </div>
  </div>

  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">Watchlist &mdash; how to read OI flow</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>OI only rises when new money opens positions.</b> Volume can churn forever between the same hands; open interest cannot. A 7-day OI build means someone is deliberately positioning &mdash; that is the whole reason the Sunday check works.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>The quadrant is price and OI together.</b> Price up + OI up = new longs (real trend). Price down + OI up = new shorts (real downtrend, or a squeeze being loaded). Price up + OI <i>down</i> is just short covering, and both-down is longs capitulating &mdash; those rows are shown dimmed because they are movement without new conviction.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Funding is the lie detector.</b> Normalised to %/day (intervals differ per contract). New longs with funding near zero or negative = spot buyers leading while shorts fight it &mdash; <b>SPOT-LED</b>, the healthy version. The same picture with rich positive funding = leverage arriving late &mdash; <b>CROWDED LONGS</b>. New shorts paying deeply negative funding = <b>SQUEEZE FUEL</b>.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Fresh listings are excluded on purpose.</b> A new contract's OI always ramps from zero, which looks like a monster build and means nothing. Pairs with under ~6 days of OI history are skipped and counted in the audit line instead of silently shown.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>HIGH OI/VOL is a crowding warning, not a direction.</b> When open interest exceeds ~1.5x average daily volume, positions cannot exit without moving the price &mdash; fuel for squeezes in either direction.</div></div>
    </div>
  </div>

  <div class="dash-sec" style="margin-top:28px">
    <div class="dash-sec-ttl">Movers &mdash; what a daily watchlist is actually screening for</div>
    <div class="rules">
      <div class="rule"><div class="rule-n">01</div><div class="rule-b"><b>Three triggers, any one qualifies.</b> A 1-day move of 5%+, <i>or</i> the day's traded value at 2x its own 20-day average, <i>or</i> the day's true range at 1.5x ATR20. A coin only has to be unusual in one way to be worth a look.</div></div>
      <div class="rule"><div class="rule-n">02</div><div class="rule-b"><b>Volume and range beat raw % change.</b> Sorting the market by 24h gainers is the lazy version of this screen: it surfaces whatever already moved and misses a dormant coin waking up on 3x volume. Both ratios are measured against each coin's <b>own</b> 20-day baseline, so a quiet large-cap and a volatile micro-cap are judged on the same scale.</div></div>
      <div class="rule"><div class="rule-n">03</div><div class="rule-b"><b>Volume is only ever a self-ratio.</b> The OHLCV volume column means different things per exchange &mdash; coins, contracts, or quote currency. Comparing one coin's day against its own average divides that unknown factor out; comparing dollar amounts across venues would not.</div></div>
      <div class="rule"><div class="rule-n">04</div><div class="rule-b"><b>Every row arrives with levels.</b> Prior-day high and low, the 7-day range boundaries, and where price sits inside that range right now. A watchlist without a level and a point where you are wrong is a list of names, not a plan &mdash; and that is the part these posts usually leave out.</div></div>
      <div class="rule"><div class="rule-n">05</div><div class="rule-b"><b>Completed bars only.</b> The scan describes the last <b>closed</b> day, so it reads the same at 02:00 and at 22:00. It tells you what happened yesterday so you can plan today; it is not a live intraday tape.</div></div>
      <div class="rule"><div class="rule-n">06</div><div class="rule-b"><b>No stocks, ETFs or commodities.</b> Those perps are excluded from the universe entirely. Their underlying market closes at the weekend, leaving a thin book, funding-driven drift and a gap at the Monday cash open &mdash; quoting a weekend level on one is quoting a level on nothing.</div></div>
      <div class="rule"><div class="rule-n">07</div><div class="rule-b"><b>The standout score is a description, not an edge.</b> Volume anomaly (35), range expansion (30), proximity to a 7-day boundary (20), size of the move (15). It has no direction and has not been backtested &mdash; the same caveat as the setup score.</div></div>
    </div>
  </div>

  <div class="dash-sec">
    <div class="dash-sec-ttl">Setup score &mdash; read this before trusting it</div>
    <div style="font-size:12px;color:var(--tx2);max-width:700px;line-height:1.75">
      The score is a <b style="color:var(--tx)">description of the structure</b>, not a win rate and not a
      backtested edge. It adds up: how large the momentum divergence is relative to the leg (30), how
      decisive the price break is in ATR terms (15), whether the pivots are sensibly spaced (15), whether
      they sit on the classic side of the MACD zero line (15), whether MACD has crossed its signal line
      since the pivot (15), and how liquid the pair is (10).
      <br><br>
      A 90 means a clean textbook picture. It does not mean the trade works. The previous version quoted
      win rates from a backtest of a different configuration &mdash; weekly regime filter on, hidden
      divergences included, 22 fixed pairs &mdash; and none of that describes what you are looking at here,
      so those numbers are gone rather than quietly repurposed.
    </div>
  </div>
</div>

</div>

<script>
var TFM = __TF_CFG_JSON__;
var cTf = '4h';

var SC = {
  active:      {icon:'&#9679;',  lbl:'At the signal', sub:'Price still near the pivot', cls:'st-active'},
  extended:    {icon:'&#9651;',  lbl:'Extended',      sub:'Already moved from the signal', cls:'st-extended'},
  ran:         {icon:'&#9650;',  lbl:'Ran',           sub:'Reached 1R or better', cls:'st-ran'},
  invalidated: {icon:'&#10005;', lbl:'Invalidated',   sub:'Traded through the stop', cls:'st-invalidated'}
};

function sbadge(d){
  var c = SC[d.status] || SC.active;
  var sub = c.sub;
  // pct_chg is the move IN THE SIGNAL'S DIRECTION, not the raw price change:
  // on a short, price falling 3% is pct_chg +3. Label it so a green +3% next
  // to a bearish setup cannot be misread as price having gone up.
  if(d.status === 'extended') sub = 'Moved ' + d.pct_chg + '% in signal direction';
  else if(d.status === 'ran') sub = 'Made ' + d.r_reached + 'R since signal';
  else if(d.status === 'active') sub = (d.pct_chg >= 0 ? '+' : '') + d.pct_chg + '% in signal direction';
  return '<div class="status ' + c.cls + '"><span class="tlbl">' + c.icon + ' ' + c.lbl + '</span>' +
         '<span class="tsub">' + sub + '</span></div>';
}
function qcolor(q){ return q >= 70 ? 'var(--bull)' : q >= 50 ? 'var(--fresh)' : 'var(--tx3)'; }
function qcell(q){
  var c = qcolor(q), lbl = q >= 70 ? 'HIGH' : q >= 50 ? 'MED' : 'LOW';
  return '<div class="qc"><span class="qnum" style="color:' + c + '">' + q +
    '<span style="font-size:9px;opacity:.7;margin-left:3px">' + lbl + '</span></span>' +
    '<div class="qbar"><div class="qfill" style="width:' + q + '%;background:' + c + '"></div></div></div>';
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
function volCell(d){
  return '<div class="volc">' + fmtVol(d.weekly_volume) + '<small>7 days</small></div>';
}
function freshTag(d){
  return d.fresh ? ' <span class="bdg bx" style="font-size:8px;padding:2px 5px">FRESH</span>' : '';
}
function b2t(bars, tf){
  var m = bars * ((TFM[tf] && TFM[tf].minutes) || 240);
  if(m < 60) return m + 'm ago';
  if(m < 1440) return Math.round(m/60) + 'h ago';
  var d = Math.round(m/1440);
  return d + (d === 1 ? ' day' : ' days') + ' ago';
}

function matches(d, f){
  if(f === 'all')    return true;
  if(f === 'live')   return d.status !== 'invalidated';
  if(f === 'fresh')  return d.fresh && d.status !== 'invalidated';
  if(f === 'active') return d.status === 'active';
  if(f === 'bull')   return d.side === 'bullish' && d.status !== 'invalidated';
  if(f === 'bear')   return d.side === 'bearish' && d.status !== 'invalidated';
  return true;
}

/* Every coin the scan did not actually analyse gets named here. The point is
   that "3 live setups" must never be readable as "the market is quiet" when
   40 coins silently failed to fetch or the scan was cut off half way. */
function auditLine(m){
  if(!m || !m.dispatched) return '';
  var bits = [];
  bits.push(m.scanned + ' of ' + m.dispatched + ' candidate pairs analysed');
  if(m.vol_rejected) bits.push(m.vol_rejected + ' below the 7-day volume floor');
  if(m.no_data)      bits.push(m.no_data + ' returned no usable candles');
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
  if(name === 'dash') loadDash();
  if(name === 'signals') loadSignals();
  if(name === 'watch') loadWatch();
  if(name === 'movers') loadMovers();
  if(name === 'calendar') loadCalendar();
}
document.querySelectorAll('.nav-btn').forEach(function(b){
  b.addEventListener('click', function(){ showPg(b.dataset.pg); });
});

/* ── Dashboard ────────────────────────────────────────────────────────────── */
function loadDash(){
  fetch('/signals').then(function(r){ return r.json(); })
    .then(renderDash)
    .catch(function(){ document.getElementById('dash-loader').textContent = 'Failed to load data.'; });
}

function renderDash(sd){
  document.getElementById('dash-loader').style.display = 'none';
  document.getElementById('dash-content').style.display = 'block';
  var sigs = sd.signals || [], meta = sd.meta || {}, ts = sd.ts;
  cTf = meta.tf || '4h';

  var live = sigs.filter(function(s){ return s.status !== 'invalidated'; });
  document.getElementById('kpi-live').textContent = live.length;
  document.getElementById('kpi-bull').textContent = live.filter(function(s){ return s.side === 'bullish'; }).length;
  document.getElementById('kpi-bear').textContent = live.filter(function(s){ return s.side === 'bearish'; }).length;
  document.getElementById('kpi-scan').textContent = meta.scanned || '-';
  document.getElementById('kpi-live-sub').textContent = sigs.length + ' total incl. invalidated';

  if(ts){
    var ma = Math.round((Date.now() - new Date(ts)) / 60000);
    var tflbl = (TFM[cTf] && TFM[cTf].label) || cTf.toUpperCase();
    document.getElementById('kpi-scan-sub').textContent =
      tflbl + ' · ' + (meta.exchange || '').toUpperCase() + ' · ' + ma + ' min ago';
  }
  document.getElementById('scan-audit').innerHTML = auditLine(meta);

  var fresh = live.filter(function(s){ return s.fresh; })
                  .sort(function(a,b){ return b.score - a.score; }).slice(0, 9);
  var rg = document.getElementById('ready-grid');
  if(fresh.length === 0){
    rg.innerHTML = '<div class="no-sig" style="grid-column:1/-1">No fresh setups in the last scan.</div>';
  } else {
    rg.innerHTML = '';
    fresh.forEach(function(s){ rg.appendChild(mkCard(s)); });
  }

  var list = document.getElementById('dash-sig-list');
  list.innerHTML = '';
  if(live.length === 0){
    list.innerHTML = '<div class="no-sig">Run a scan from the Scanner tab.</div>';
  } else {
    live.sort(function(a,b){ return b.score - a.score; })
        .forEach(function(s){ list.appendChild(mkRow(s)); });
  }
}

function mkCard(s){
  var c = SC[s.status] || SC.active;
  var isBull = s.side === 'bullish';
  var qc = qcolor(s.score);
  var tgt = s.targets || {};
  var div = document.createElement('div');
  div.className = 'sig-card';
  div.innerHTML =
    '<div class="sc-timing ' + c.cls + '">' + c.icon + ' ' + c.lbl + '</div>' +
    '<div class="sc-body">' +
      '<div class="sc-header"><span class="sc-pair">' + s.base + '</span>' +
        '<span class="bdg ' + (isBull ? 'bb' : 'br') + '" style="font-size:9px">' + s.type + '</span>' + freshTag(s) + '</div>' +
      '<div class="sc-meta">' +
        '<div><div class="sml">Weekly volume</div><div class="smv">' + fmtVol(s.weekly_volume) + '</div></div>' +
        '<div><div class="sml">In signal direction</div><div class="smv ' + (s.pct_chg >= 0 ? 'bull' : 'bear') + '">' + (s.pct_chg >= 0 ? '+' : '') + s.pct_chg + '% <span style="color:var(--tx3);font-size:9px">(px ' + (s.price_chg_pct >= 0 ? '+' : '') + s.price_chg_pct + '%)</span></div></div>' +
        '<div><div class="sml">Signal date</div><div class="smv" style="font-size:9px">' + s.pivot2_time + '</div></div>' +
        '<div><div class="sml">3R target</div><div class="smv ' + (isBull ? 'bull' : 'bear') + '">' + fmtPx(tgt['3r']) + '</div></div>' +
      '</div>' +
      '<div class="qbar-wrap"><div class="qbar-lbl"><span>Setup score</span><span style="color:' + qc + '">' + s.score + '/100</span></div>' +
        '<div class="qbar-bg"><div class="qbar-fill" style="width:' + s.score + '%;background:' + qc + '"></div></div></div>' +
    '</div>';
  return div;
}

function mkRow(s){
  var isBull = s.side === 'bullish';
  var tgt = s.targets || {};
  var div = document.createElement('div');
  div.className = 'srow' + (s.status === 'invalidated' ? ' dim' : '');
  div.innerHTML =
    '<div class="srow-pair"><div style="display:flex;align-items:center;gap:5px">' +
      '<span style="font-family:var(--mono);font-size:14px;font-weight:600;color:var(--tx)">' + s.base + '</span>' + freshTag(s) + '</div>' +
      '<span style="font-family:var(--mono);font-size:9px;color:var(--tx3)">USDT PERP</span></div>' +
    '<div>' + sbadge(s) + '</div>' +
    '<div><span class="bdg ' + (isBull ? 'bb' : 'br') + '">' + s.type + '</span></div>' +
    '<div>' + qcell(s.score) + '</div>' +
    '<div>' + volCell(s) + '</div>' +
    '<div class="ptm">' + s.pivot2_time + '<br>' + b2t(s.bars_ago, s.tf || cTf) + '</div>' +
    '<div class="tgt-row">' +
      '<span style="color:var(--bear)">Stop ' + fmtPx(s.stop_price) + ' (risk ' + s.risk_pct + '%)</span>' +
      '<span style="color:var(--tx3)">|</span>' +
      '<span style="color:var(--tx2)">1R ' + fmtPx(tgt['1r']) + '</span>' +
      '<span style="color:var(--tx2)">2R ' + fmtPx(tgt['2r']) + '</span>' +
      '<span style="color:var(--bull);font-weight:600">3R ' + fmtPx(tgt['3r']) + '</span>' +
    '</div>';
  return div;
}

/* ── Signals page ─────────────────────────────────────────────────────────── */
var _sf = 'live', _sigs = [];
function loadSignals(){
  fetch('/signals').then(function(r){ return r.json(); }).then(function(d){
    _sigs = d.signals || [];
    cTf = (d.meta || {}).tf || cTf;
    var ts = d.ts, ma = ts ? Math.round((Date.now() - new Date(ts)) / 60000) + ' min ago' : '-';
    document.getElementById('sig-sub').textContent =
      _sigs.length + ' signals · ' + ((d.meta || {}).scanned || 0) + ' pairs analysed · ' + ma;
    renderSigs();
  });
}
function setSF(btn){
  document.querySelectorAll('.sig-ftab').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active'); _sf = btn.dataset.sf; renderSigs();
}
function renderSigs(){
  var s = _sigs.filter(function(x){ return matches(x, _sf); });
  var c = document.getElementById('sigList');
  if(s.length === 0){ c.innerHTML = '<div class="no-sig">No signals match this filter.</div>'; return; }
  c.innerHTML = '';
  s.sort(function(a,b){ return b.score - a.score; });
  s.forEach(function(x){ c.appendChild(mkRow(x)); });
}

/* ── Watchlist (OI flow) ──────────────────────────────────────────────────── */
var _wf = 'all', _wrows = [], wes = null;
var WQ = {
  'NEW LONGS':      {cls:'bdg bb'},
  'NEW SHORTS':     {cls:'bdg br'},
  'SHORT COVERING': {cls:'wq wq-sc'},
  'CAPITULATION':   {cls:'wq wq-cap'}
};
var WFLAG = {'SPOT-LED':'wf-spot','CROWDED LONGS':'wf-crowd','SQUEEZE FUEL':'wf-squeeze','HIGH OI/VOL':'wf-hioi'};

var TH = {chg:__MOVER_CHG__, vol:__MOVER_VOL__, rng:__MOVER_RNG__, edge:__MOVER_EDGE__,
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
  'BELOW 7D LOW':    'Price has traded through the 7-day low.'
};
function tip(k){ return TIP[k] ? ' title="' + TIP[k] + '"' : ''; }

function wMatches(r, f){
  if(f === 'longs')   return r.quadrant === 'NEW LONGS';
  if(f === 'shorts')  return r.quadrant === 'NEW SHORTS';
  if(f === 'squeeze') return (r.flags || []).indexOf('SQUEEZE FUEL') >= 0;
  if(f === 'spot')    return (r.flags || []).indexOf('SPOT-LED') >= 0;
  return true;
}
function setWF(btn){
  document.querySelectorAll('.sig-ftab[data-wf]').forEach(function(b){ b.classList.remove('active'); });
  btn.classList.add('active'); _wf = btn.dataset.wf; renderWatch();
}
function fmtPct(v){
  if(v === null || v === undefined) return '&mdash;';
  return (v >= 0 ? '+' : '') + v + '%';
}
function watchAudit(m){
  if(!m || !m.dispatched) return '';
  var bits = [m.analysed + ' of ' + m.dispatched + ' pairs measured',
              (m.quiet || 0) + ' quiet (OI moved < 5%)'];
  if(m.vol_rejected) bits.push(m.vol_rejected + ' below the volume floor');
  if(m.no_oi)        bits.push(m.no_oi + ' had no usable OI history (fresh listings excluded on purpose)');
  if(m.errors)       bits.push(m.errors + ' errored');
  var txt = bits.join(' · ');
  if(m.complete === false){
    return '<div class="audit warn">&#9888; This OI scan did not finish &mdash; re-run before trusting it.<br>' + txt + '</div>';
  }
  return txt;
}
function renderWatch(){
  var rows = _wrows.filter(function(r){ return wMatches(r, _wf); });
  rows.sort(function(a, b){ return b.oi_chg_pct - a.oi_chg_pct; });
  var tb = document.getElementById('wB'); tb.innerHTML = '';
  if(rows.length === 0){
    document.getElementById('wT').style.display = 'none';
    document.getElementById('wEmpty').style.display = '';
    document.getElementById('wEmpty').textContent = _wrows.length ?
      'No pairs match this filter.' : 'No OI scan yet - hit Scan OI Flow. Sunday is the classic day to look.';
    return;
  }
  document.getElementById('wEmpty').style.display = 'none';
  document.getElementById('wT').style.display = 'table';
  rows.forEach(function(r){
    var q = WQ[r.quadrant] || WQ['CAPITULATION'];
    var dimmed = (r.quadrant === 'SHORT COVERING' || r.quadrant === 'CAPITULATION');
    var flags = (r.flags || []).map(function(f){
      return '<span class="wflag ' + (WFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>';
    }).join('') || '<span style="color:var(--tx3)">&mdash;</span>';
    var fund = r.funding_day_pct === null || r.funding_day_pct === undefined ? '&mdash;'
             : (r.funding_day_pct >= 0 ? '+' : '') + r.funding_day_pct.toFixed(3) + '%';
    var tr = document.createElement('tr');
    if(dimmed) tr.className = 'dim';
    tr.innerHTML =
      '<td><div class="pb"><span class="pb-pair">' + r.base + '</span><span class="pb-q">' + fmtPx(r.current_price) + '</span></div></td>' +
      '<td><span style="font-family:var(--mono);font-weight:600;color:' + (r.oi_chg_pct >= 0 ? 'var(--bull)' : 'var(--bear)') + '">' + fmtPct(r.oi_chg_pct) + '</span></td>' +
      '<td class="volc">' + fmtVol(r.oi_usd) + (r.oi_vol_ratio ? '<small>' + r.oi_vol_ratio + 'x daily vol</small>' : '') + '</td>' +
      '<td><span style="font-family:var(--mono);color:' + (r.price_chg_pct >= 0 ? 'var(--bull)' : 'var(--bear)') + '">' + fmtPct(r.price_chg_pct) + '</span></td>' +
      '<td><span class="' + q.cls + '"' + tip(r.quadrant) + '>' + r.quadrant + '</span></td>' +
      '<td><span style="font-family:var(--mono);font-size:11px;color:var(--tx2)">' + fund + '</span></td>' +
      '<td>' + flags + '</td>' +
      '<td>' + volCell(r) + '</td>';
    tb.appendChild(tr);
  });
}
function loadWatch(){
  fetch('/watch_signals').then(function(r){ return r.json(); }).then(function(d){
    _wrows = d.rows || [];
    var m = d.meta || {};
    if(d.ts){
      var ma = Math.round((Date.now() - new Date(d.ts)) / 60000);
      document.getElementById('watch-sub').textContent =
        _wrows.length + ' movers on ' + (m.exchange || '').toUpperCase() + ' · ' +
        (m.lookback_d || 7) + '-day OI change · scanned ' + ma + ' min ago';
      if(m.exchange) document.getElementById('wExchange').value = m.exchange;
    }
    document.getElementById('watch-audit').innerHTML = watchAudit(m);
    renderWatch();
  }).catch(function(){});
}
function startWatch(){
  if(wes){ wes.close(); wes = null; }
  var ex  = document.getElementById('wExchange').value;
  var vol = (parseInt(document.getElementById('minVol').value) || 70) * 1e6;
  var spd = document.querySelector('#spdG .tbtn.active').dataset.val;
  var mp  = parseInt(document.getElementById('maxPairs').value) || 400;
  var btn = document.getElementById('wRunBtn');
  btn.disabled = true;
  _wrows = [];
  renderWatch();
  document.getElementById('watch-audit').innerHTML = '';
  var wtot = 0, prog = document.getElementById('wProg');
  prog.textContent = 'Loading markets...';

  wes = new EventSource('/watchlist?exchange=' + ex + '&min_weekly_vol=' + vol +
                        '&speed=' + spd + '&max_pairs=' + mp);
  wes.addEventListener('log', function(e){ prog.textContent = JSON.parse(e.data).msg; });
  wes.addEventListener('total', function(e){ wtot = JSON.parse(e.data).total; });
  wes.addEventListener('progress', function(e){
    var d = JSON.parse(e.data);
    prog.textContent = d.current + '/' + wtot + ' - ' + d.pair.replace('/USDT:USDT','');
  });
  wes.addEventListener('result', function(e){
    _wrows.push(JSON.parse(e.data));
    renderWatch();
  });
  wes.addEventListener('done', function(e){
    var d = JSON.parse(e.data);
    wes.close(); btn.disabled = false;
    prog.textContent = 'Done - ' + d.hits + ' mover(s)';
    document.getElementById('watch-audit').innerHTML = watchAudit(d.meta || {});
    var m = d.meta || {};
    document.getElementById('watch-sub').textContent =
      d.hits + ' movers on ' + (m.exchange || '').toUpperCase() + ' · ' +
      (m.lookback_d || 7) + '-day OI change · just now';
  });
  wes.addEventListener('error', function(e){
    if(!e.data) return;
    var m = 'Scan error';
    try { m = JSON.parse(e.data).msg; } catch(x){}
    prog.textContent = m; wes.close(); btn.disabled = false;
  });
  wes.onerror = function(){
    if(wes.readyState === EventSource.CLOSED) return;
    // same reasoning as the divergence scan: an EventSource reconnect would
    // silently launch a SECOND full OI scan. Close and let the user re-run.
    wes.close(); btn.disabled = false;
    prog.textContent = 'Connection lost - scan stopped.';
  };
}

/* ── Movers (daily watchlist) ─────────────────────────────────────────────── */
var _mf = 'all', _mrows = [], mes = null;
var MFLAG = {'VOLUME SPIKE':'mf-vol','RANGE EXPANSION':'mf-rng','BIG MOVE':'mf-move',
             'AT RANGE HIGH':'mf-hi','ABOVE 7D HIGH':'mf-hi',
             'AT RANGE LOW':'mf-lo','BELOW 7D LOW':'mf-lo'};

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
  var txt = bits.join(' · ');
  if(m.complete === false){
    return '<div class="audit warn">&#9888; This movers scan did not finish &mdash; re-run before trusting it.<br>' + txt + '</div>';
  }
  return txt;
}
function renderMovers(){
  var rows = _mrows.slice().sort(function(a, b){ return b.score - a.score; })
                   .filter(function(r){ return mMatches(r, _mf); });
  if(_mf === 'top') rows = rows.slice(0, 4);
  var tb = document.getElementById('mvB'); tb.innerHTML = '';
  if(rows.length === 0){
    document.getElementById('mvT').style.display = 'none';
    var e = document.getElementById('mvEmpty');
    e.style.display = '';
    e.textContent = _mrows.length ? 'No pairs match this filter.' : 'No movers scan yet - hit Scan Movers.';
    return;
  }
  document.getElementById('mvEmpty').style.display = 'none';
  document.getElementById('mvT').style.display = 'table';
  rows.forEach(function(r, i){
    var flags = (r.flags || []).map(function(f){
      return '<span class="wflag ' + (MFLAG[f] || 'wf-hioi') + '"' + tip(f) + '>' + f + '</span>';
    }).join('') || '<span style="color:var(--tx3)">&mdash;</span>';
    var up  = r.chg_1d_pct >= 0;
    var big = Math.abs(r.chg_1d_pct || 0) >= TH.chg;
    var tr = document.createElement('tr');
    tr.className = up ? 'rb' : 'rr';
    tr.innerHTML =
      '<td><span class="mrank' + (i < 4 ? ' top' : '') + '">' + (i + 1) + '</span></td>' +
      '<td><div class="pb"><span class="pb-pair">' + r.base + '</span><span class="pb-q">' + fmtPx(r.current_price) + '</span></div></td>' +
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
    tb.appendChild(tr);
  });
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

  mes = new EventSource('/movers?exchange=' + ex + '&min_weekly_vol=' + vol +
                        '&speed=' + spd + '&max_pairs=' + mp);
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
  loose:    'Only requires price and MACD to disagree between two pivots. Most signals, including marginal ones you would reject by eye.',
  balanced: 'MACD pivot must align with the price pivot, the divergence line must be unbroken, and the move must clear a minimum size.',
  strict:   'Balanced plus the zero-line condition: both bullish MACD pivots below zero, both bearish above. Fewer signals, higher conviction.'
};
['tfG','strG','spdG'].forEach(function(g){
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
function updStats(side, fresh){
  if(side === 'bullish'){ cb++; bump(document.getElementById('cBull'), cb); document.getElementById('stBull').classList.add('vis'); }
  else { cr++; bump(document.getElementById('cBear'), cr); document.getElementById('stBear').classList.add('vis'); }
  if(fresh){ cf++; bump(document.getElementById('cFresh'), cf); document.getElementById('stFresh').classList.add('vis'); }
}
function resetStats(){
  cb = cr = cf = 0;
  ['stBull','stBear','stFresh'].forEach(function(i){ document.getElementById(i).classList.remove('vis'); });
  ['cBull','cBear','cFresh'].forEach(function(i){ document.getElementById(i).textContent = '0'; });
}

var af = 'live', sortKey = 'score', pendR = [], hasScanned = false, seenSig = {};
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

function buildRow(d){
  var bull = d.side === 'bullish';
  var base = d.base || (d.pair || '').replace('/USDT:USDT','').replace('/USDT','');
  var tgt = d.targets || {};
  var fromCur = d.targets_from === 'current';
  var tr = document.createElement('tr');
  tr.className = (bull ? 'rb' : 'rr') + (d.status === 'invalidated' ? ' dim' : '');
  var basis = '<div style="font-family:var(--mono);font-size:8px;color:' + (fromCur ? 'var(--fresh)' : 'var(--tx3)') + ';margin-bottom:2px">' +
    (fromCur ? 'FROM CURRENT PRICE ' + fmtPx(d.entry_ref_price) : 'FROM SIGNAL PRICE') + '</div>';
  var tgts = basis + '<span style="color:var(--tx3)">' + fmtPx(tgt['1r']) + '</span>' +
    ' <span style="color:var(--tx3)">/</span> <span style="color:var(--tx2)">' + fmtPx(tgt['2r']) + '</span>' +
    ' <span style="color:var(--tx3)">/</span> <span style="color:var(--bull);font-weight:600">' + fmtPx(tgt['3r']) + '</span>';
  tr.innerHTML =
    '<td><div class="pb"><div style="display:flex;align-items:center;gap:5px"><span class="pb-pair">' + base + '</span>' + freshTag(d) + '</div>' +
      '<span class="pb-q">USDT PERP</span></div></td>' +
    '<td>' + sbadge(d) + '</td>' +
    '<td><span class="bdg ' + (bull ? 'bb' : 'br') + '">' + d.type + '</span></td>' +
    '<td>' + qcell(d.score || 0) + '</td>' +
    '<td>' + volCell(d) + '</td>' +
    '<td><div class="ptm" style="color:var(--tx);font-size:11px">' + d.pivot2_time + '</div><div class="ptm">' + b2t(d.bars_ago, d.tf || cTf) + '</div></td>' +
    '<td><span style="font-family:var(--mono);font-size:11px;color:var(--bear)">' + fmtPx(d.stop_price) + '</span><div style="font-family:var(--mono);font-size:9px;color:var(--tx3)">risk ' + d.risk_pct + '%</div></td>' +
    '<td style="font-family:var(--mono);font-size:10px">' + tgts + '</td>';
  return tr;
}

function renderAll(results){
  var rows = (results || []).filter(function(d){ return matches(d, af); });
  rows.sort(function(a, b){
    if(sortKey === 'vol') return (b.weekly_volume || 0) - (a.weekly_volume || 0);
    if(sortKey === 'age') return a.bars_ago - b.bars_ago;
    return b.score - a.score;
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
    em.innerHTML = '<span class="empty-icon">&#9676;</span><span>No signals match this filter</span>';
  } else {
    document.getElementById('emS').style.display = 'none';
    document.getElementById('resT').style.display = 'table';
  }
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

  var exch = document.getElementById('exchange').value;
  var tf   = document.querySelector('#tfG .tbtn.active').dataset.val;
  var str  = document.querySelector('#strG .tbtn.active').dataset.val;
  var spd  = document.querySelector('#spdG .tbtn.active').dataset.val;
  var vol  = (parseInt(document.getElementById('minVol').value) || 70) * 1e6;
  var mp   = parseInt(document.getElementById('maxPairs').value) || 400;
  var syms = document.getElementById('symOverride').value.trim();
  cTf = tf;

  var url = '/scan?exchange=' + exch + '&tf=' + tf + '&strictness=' + str +
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
    var k = d.pair + '|' + d.side + '|' + d.pivot2_time;
    if(seenSig[k]) return;            // duplicate guard, see es.onerror
    seenSig[k] = 1;
    updStats(d.side, d.fresh);
    pendR.push(d);
    pushBg(d.pair, true);
  });
  es.addEventListener('done', function(e){
    var d = JSON.parse(e.data);
    es.close();
    document.getElementById('runBtn').disabled = false;
    setProgress(tot, tot, '');
    setStatus('Done - ' + d.hits + ' signal' + (d.hits !== 1 ? 's' : ''), false);
    var m = d.meta || {};
    pushLog('Scan complete - ' + d.hits + ' signal(s) from ' + (m.scanned || 0) + ' pairs analysed', 'dn');
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
    if(pendR.length){ dissolve(); setTimeout(function(){ renderAll(pendR); }, 350); }
  };
}

function renderWelcome(){
  try {
    var el = document.getElementById('emS');
    if(!el) return;
    var h = new Date().getHours();
    var gr = h < 12 ? 'Good morning' : h < 18 ? 'Good afternoon' : 'Good evening';
    var subs = ['Ready to find divergences.',
                'Full market, regular divergences only.',
                'Every liquid perp, one scan.',
                'Configure the volume floor and hit Run.'];
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
    var msg = _cal.events.length ? 'No market movers on the schedule through ' + calDayLabel(new Date(_cal.events[_cal.events.length - 1].ts)) + '.' : 'Calendar not loaded yet.';
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
  /* Past the last day either feed has published, a blank cell means "no data", not "quiet". Computed on all events, not the filtered ones. */
  var lastDay = _cal.events.length ? calDayKey(new Date(_cal.events[_cal.events.length - 1].ts)) : '';
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
      var nodata = lastDay && k > lastDay;
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
  if(_cal.events.length) bits.push('through ' + calDayLabel(new Date(_cal.events[_cal.events.length - 1].ts)));
  document.getElementById('cal-meta').textContent = bits.join(' \u00b7 ');
  var warn = [], info = [];
  if(m.fallback) warn.push(m.fallback + ': only the current week is shown. A working JBlanked key in .env extends the horizon.');
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
  loadDash();
  loadCalendar();
  var pg = (location.hash || '').slice(1);
  if(pg && document.getElementById('pg-' + pg)) showPg(pg);   /* deep link, e.g. /#calendar */
});
</script>
</body>
</html>"""

if __name__ == "__main__":
    load_dotenv()     # JBLANKED_API_KEY from .env, if there is one
    load_state()      # restore the last scan so the dashboard isn't amnesiac
    load_movers()     # ...and the last movers scan
    load_calendar()   # ...and the news calendar
    app.run(debug=False, threaded=True, port=5099)
