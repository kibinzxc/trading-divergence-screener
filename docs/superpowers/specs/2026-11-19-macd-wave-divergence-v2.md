# MACD Divergence Screener: specification v2

## What changed from v1 (why)
v1 found divergences from PRICE pivots (5 left / 3 right) and then looked for a nearby
MACD value. That matches any two price lows, which is the first chart I made, not
the trough-to-trough read on the MACD lines.
v2 reads divergence the way you do on TradingView:
1. Split the MACD line into waves using the MACD/signal crossovers (the dots).
2. Each wave below the signal line is one TROUGH, and each wave above it is one PEAK.
3. Compare consecutive troughs (or peaks) of the MACD line with the price extremes
   inside those same waves.
Because the two troughs are separated by a bullish then a bearish cross, the W shape
(the bump in the middle) is built into the detection.

## Purpose
Scans crypto perpetual futures for REGULAR MACD-line divergences on 4H and 1H, using
MACD crossover waves. For each setup it gives a quality score and says whether it is
good to enter right now.
Hidden divergences are not detected. The histogram is not used for detection.

## Universe (unchanged)
- Exchanges: OKX, Binance, Bybit (via ccxt). Linear USDT perpetuals only.
- Excluded: stablecoin bases, and stock / ETF / commodity perps.
- Liquidity gate: rolling 7-day traded value >= $70M (adjustable), from 4H candles.
- Most liquid first, up to 400 pairs.

## Indicators
- MACD 12 / 26 / 9 on close (EMA). Divergence is read on the MACD LINE.
- ATR 14 (Wilder). EMA 200 on each timeframe. RSI 14 on BTC 4H.
- Only completed candles. The still-forming candle is dropped, so nothing repaints.
- History: 720 bars of 4H, 500 bars of 1H, plus 200 bars of 1D for higher-timeframe checks.

## Step 1: Crossovers
- Bullish cross at bar i: MACD[i-1] <= signal[i-1] and MACD[i] > signal[i].
- Bearish cross at bar i: MACD[i-1] >= signal[i-1] and MACD[i] < signal[i].
- Whipsaw filter: a cross counts only if the gap between MACD and signal after it reaches
  >= 5% of the 50-bar MACD range within 3 bars. Otherwise it is merged with the wave
  it interrupts. This stops tiny double dots from splitting one trough into two.

## Step 2: Waves
- A BEAR WAVE runs from a bearish cross to the next bullish cross.
  A BULL WAVE runs from a bullish cross to the next bearish cross.
- Minimum wave length: 3 bars. Shorter waves are merged into the neighbouring wave.
- A wave is closed only when its ending cross has printed on a completed candle.

## Step 3: Troughs and peaks (one per wave)
- MACD TROUGH of a bear wave = the lowest MACD-line value in that wave (bar tM).
- PRICE LOW of that wave = the lowest LOW (wick) from the wave start to 2 bars after
  the wave end (bar tP). The extra 2 bars catch a price low that prints just after
  the MACD turns.
- Bull waves work the same way with the MACD PEAK and the highest HIGH.

## Step 4: Divergence (troughs A then B)
- Bullish: two bear waves A and B, with exactly one bull wave (the bump) between them,
  or at most two bull waves when using the extended mode (off by default).
  - Price low B < price low A.
  - MACD trough B > MACD trough A.
- Bearish: the mirror image, using bull waves, price highs and MACD peaks.
- Spacing: A's MACD trough and B's MACD trough are 8 to 80 bars apart.
- Bump height (the middle of the W): the highest MACD value in the bull wave between
  them must be at least 25% of the leg range above the higher of the two troughs.
  Leg range = max(MACD) - min(MACD) from A to B.
- Minimum size (Balanced, default):
  - Price: (A low - B low) / A low >= max(0.25%, 0.20 x ATR%).
  - MACD: (B trough - A trough) >= 15% of the leg range.
- Strict preset: 0.50%, 0.55 x ATR%, 22%, and both troughs below zero for bullish
  (both peaks above zero for bearish).
- Triple divergence: if trough C before A also forms a valid divergence with A
  (lower price, higher MACD), tag it TRIPLE.

## Step 5: Signal bar
- The signal bar is the bullish cross that ENDS wave B (or the bearish cross for bearish).
- Signal price = the close of the signal bar.
- So a divergence is only reported once wave B is closed by a cross. This replaces
  rule 5 of v1 ("MACD has not crossed since pivot B").

## Step 6: Quality score (0 to 100)
| Component | Points |
|---|---|
| Price difference A to B, as a multiple of ATR (capped at 1.0 x ATR) | 0-15 |
| MACD trough improvement, as % of leg range (capped at 50%) | 0-15 |
| Bump height above the 25% minimum (capped at 75%) | 0-10 |
| Both troughs on the classic side of zero | 10 |
| Higher-timeframe agreement: 1H signal needs the 4H MACD rising; 4H signal needs the 1D MACD rising (falling for bearish) | 15 |
| Volume at B below volume at A (average over each wave), which shows sellers drying up | 10 |
| Rejection wick at the B price extreme: lower wick >= 50% of the candle range (upper for bearish) | 10 |
| Open interest: for bullish, OI falling into B with funding <= 0; for bearish, OI falling with funding >= 0 | 10 |
| Triple divergence | 5 |
- Penalty: -10 when OI rose more than 5% into B against the signal (for bullish, with funding > 0).
- Grades: A >= 70, B 55-69, C < 55. Only A and B grades can become "Yes".

## Step 7: Regime tag (changes the cutoff, never blocks)
- WITH TREND: bullish when BTC 4H RSI > 50 and the coin is above its 200 EMA on the
  signal timeframe (bearish is the mirror).
- COUNTER TREND: otherwise. A counter-trend setup needs a grade of A (>= 70) to show "Yes".

## Step 8: Lower-timeframe trigger (early entry)
- For a 4H divergence, look at the 1H from the 4H signal bar onward. Triggered when the
  1H MACD makes a bullish cross AND a 1H candle closes above the last 1H swing high
  since 4H price low B (3/3 pivots).
- For a 1H divergence, use the 15m the same way.
- The trigger is an early entry. The 4H signal bar itself is the late, confirmed entry.

## Status (judged at the live price)
- invalidated: price traded through the stop.
- ran: price already reached 1R or more in the signal direction.
- extended: price moved more than 2% in the signal direction, but less than 1R.
- active: still near the signal price.

## Stop and targets
- Stop: beyond price low B (or price high B for bearish) by 0.25 x ATR.
- Entry reference: the signal price, or the trigger price if a lower-timeframe trigger fired first.
- R = distance from the entry reference to the stop (minimum 0.5% of price).
- Targets: 1R, 2R, 3R, plus a structure target at the highest MACD bump bar's price high
  (lowest low for bearish).

## "Good to enter?" (first matching rule wins)
1. Stop was hit -> No.
2. Already ran 1R or more -> No.
3. Signal bar too old: more than 6 bars on 4H, or more than 12 bars on 1H -> No.
4. Grade C -> No (weak).
5. Counter trend and grade below A -> Not yet (watch only).
6. Extended (moved more than 2%) -> Not yet, wait for a pullback to the signal price.
7. A major macro release is due within 24 hours -> Not yet.
8. Otherwise -> Yes. Show "Yes (early)" if only the lower-timeframe trigger has fired
   and the signal bar is still forming, and "Yes (confirmed)" once the signal bar is closed.
This is a checklist, not a prediction.

## Backtest mode (new)
- Rerun Steps 1-8 bar by bar over the stored history with no lookahead.
- For each signal record: grade, regime, timeframe, whether 1R / 2R / 3R was hit
  before the stop, bars to outcome, and the maximum adverse move.
- Report win rate and average R per grade and per regime. Use this to tune the score cutoffs.

## Rows
- One row per coin and side. If 4H and 1H diverge the same way, they share a row.
- The row shows the timeframe with the better answer (Yes, then Not yet, then No).
  4H wins a tie, then the higher score.
- Order: Yes, then Not yet, then No. Within each group, highest score first, and "No" rows newest first.
- Columns: coin, side, TF, grade/score, regime, status, answer, signal price, stop,
  1R / 2R / 3R, A and B dates, tags.

## Context tags (shown only, never change the answer) (unchanged, plus new)
- Daily movers: VOLUME SPIKE, RANGE EXPANSION, BIG MOVE, AT RANGE HIGH / LOW,
  ABOVE 7D HIGH / BELOW 7D LOW.
- OI flow with funding: NEW LONGS, NEW SHORTS, SHORT COVERING, CAPITULATION, SPOT-LED,
  CROWDED LONGS, SQUEEZE FUEL, HIGH OI/VOL.
- New: TRIPLE, W-PATTERN (the bump >= 50%), LTF TRIGGER, ZERO-LINE CROSS
  (the MACD line has already crossed zero after the signal).

## Chart preview (new)
- Clicking a row draws price and MACD for the A-to-B leg: trough dots, crossover dots,
  the A-B line on price and on MACD, the stop, and the 1R / 2R / 3R lines.

## Other tabs (unchanged)
- Movers, Calendar.

## Implementation
One Python file: Flask, ccxt, pandas, numpy (matplotlib for the chart preview). Results
stream to a browser page. Detection lives in one function, detect_wave_divergence(df, side, preset),
so the live scan and the backtest share the same code.
