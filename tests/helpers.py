"""Synthetic candles whose MACD line is built by hand, one lobe per wave."""
import numpy as np
import pandas as pd


def wave_df(lobes, side="bullish", freq="1h", offset=0.0):
    """lobes: (bars, MACD amplitude, price depth), alternating, the first one a bump.

    For side="bullish" the odd lobes are troughs below a flat signal line and
    price dips by `depth` inside them. side="bearish" is the same picture
    upside down. `offset` lifts MACD and signal together (zero-line tests).
    """
    sgn = 1 if side == "bullish" else -1
    macd, ext = [], []
    for k, (n, amp, depth) in enumerate(lobes):
        shape = np.sin(np.pi * (np.arange(n) + 0.5) / n)
        bump = k % 2 == 0
        macd.append((amp if bump else -amp) * shape)
        ext.append(np.zeros(n) if bump else -depth * shape)
    macd = sgn * np.concatenate(macd) + offset
    ext = 100 + sgn * np.concatenate(ext)
    low, high = (ext, ext + 1) if side == "bullish" else (ext - 1, ext)
    idx = pd.date_range("2026-01-01", periods=len(macd), freq=freq)
    return pd.DataFrame({"open": low + .5, "high": high, "low": low, "close": low + .5,
                         "volume": 1.0, "macd": macd, "signal": offset, "atr": 1.0,
                         "ema200": 100.0}, index=idx)


# Trough A (MACD -2, price -5), bump, trough B (MACD -1, price -8), closing cross at bar 44.
W = [(10, 1, 0), (12, 2.0, 5), (10, 1.2, 0), (12, 1.0, 8), (8, 1, 0)]
# The same picture after 60 quiet bars: 102 bars, closing cross at bar 94.
LONG = [(60, 1, 0)] + W[1:]
