#!/usr/bin/env python3
"""FAA outage-attribution price forecaster (Python 3.13, stdlib only).

Forecasts the Yes price of every market in a case for the 240 minutes
after its checkpoint, five quantiles per minute.

Model, per market -- jump-aware random walk with a slow pull home:

    q(tau, h) = clamp(mid(h) + k(tau) * vol * sqrt(h), 0, 1)

  * vol -- per-market spread, from the size of recent 1-minute moves.
    Prices mostly sit still, then tick or jump, so the raw median move
    is 0 and tells us nothing. Instead we take the mean absolute
    1-minute change, each minute's move clipped at CLIP so a single
    huge jump cannot inflate the spread, averaged over a short window
    (120 minutes, so the spread tracks the market's current mood) and
    a long window (720 minutes) and blended 50/50, then converted
    from mean-absolute to standard-deviation units with the normal
    factor 1.2533 and bounded by a floor and a cap.
  * mid(h) -- the checkpoint price, drifting linearly over the horizon
    toward the market's recent median price (a price that just jumped
    far from where it has been living tends to drift back). The total
    drift is a fraction of the gap, capped in absolute terms.
  * k(tau) -- heavy-tailed quantile multipliers, wider in the tails
    than a normal distribution because jumps happen: the 5%/95%
    bands sit about 52% wider than normal, the 25%/75% bands about
    the same as normal.
  * Markets too young to estimate from (fewer than MIN_LEN prices)
    fall back to the house baseline: vol 0.0025, no drift. A market
    with no prices at all is centred at 0.5 with the same fallback
    spread, so the program never crashes on an odd case.

Reads every *.in file in the input directory (name order) and writes
one *.out file per case to the output directory. Directories come from
$MAKEFUN_INPUT_DIR / $MAKEFUN_OUTPUT_DIR, else /input / /output.
Deterministic: no randomness, no clock, no network, no environment
reads beyond the two directory variables.
"""

import json
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Heavy-tailed multipliers for the five levels (see module docstring).
K = [-2.5, -0.70, 0.0, 0.70, 2.5]

# --- volatility estimator -------------------------------------------------
SHORT_WIN = 120       # minutes of recent history for the short estimate
LONG_WIN = 720        # minutes of history for the long estimate
SHORT_SHARE = 0.5     # weight of the short estimate in the blend
CLIP = 0.02           # cap on one minute's |move| before averaging
ABS_TO_SD = 1.2533    # E|N(0,s)| = s*sqrt(2/pi), so s = mean|.| * 1.2533
VOL_MIN = 0.0002      # floor on vol (a flat market still wobbles)
VOL_MAX = 0.02        # cap on vol (a wild market is not infinitely wild)
VOL_DEFAULT = 0.0025  # house baseline spread, used for young markets
MIN_LEN = 10          # markets shorter than this use the fallback

# --- drift toward the recent median ---------------------------------------
ANCHOR_WIN = 480      # minutes whose median price is the anchor
DRIFT_SHARE = 0.25    # fraction of (anchor - last) drifted over
DRIFT_MAX = 0.03      # largest total drift, in price units
DRIFT_EXP = 1.0       # phase-in shape: drift grows as (h/horizon)**DRIFT_EXP
SPREAD_EXP = 0.5      # spread grows as h**SPREAD_EXP (0.5 = random walk)


def _round6(x):
    """Clamp to [0, 1], round to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1_000_000) / 1_000_000
    return v + 0.0


def _middle(values):
    """Median of an already-sorted list."""
    n = len(values)
    if n == 0:
        return 0.0
    half = n // 2
    if n % 2:
        return values[half]
    return 0.5 * (values[half - 1] + values[half])


def _clipped_mean_move(prices, span):
    """Mean |1-min move| over the last `span` minutes, each move clipped.

    `span` counts minutes of history; we need span+1 prices for span
    moves, and simply use what we have when the series is shorter.
    """
    tail = prices[-span - 1:] if len(prices) > span + 1 else prices
    if len(tail) < 2:
        return 0.0
    total = 0.0
    for i in range(len(tail) - 1):
        step = abs(tail[i + 1] - tail[i])
        total += CLIP if step > CLIP else step
    return total / (len(tail) - 1)


def _volatility(prices):
    """Per-market spread in price units per square-root minute."""
    short = _clipped_mean_move(prices, SHORT_WIN)
    long_ = _clipped_mean_move(prices, LONG_WIN)
    vol = ABS_TO_SD * (SHORT_SHARE * short + (1.0 - SHORT_SHARE) * long_)
    if vol < VOL_MIN:
        return VOL_MIN
    if vol > VOL_MAX:
        return VOL_MAX
    return vol


def _total_drift(prices, last):
    """Total pull from the checkpoint price toward the recent median."""
    anchor = _middle(sorted(prices[-ANCHOR_WIN:]))
    drift = DRIFT_SHARE * (anchor - last)
    if drift > DRIFT_MAX:
        return DRIFT_MAX
    if drift < -DRIFT_MAX:
        return -DRIFT_MAX
    return drift


def market_rows(market, horizon):
    """The 240 x 5 quantile rows for one market."""
    prices = market["prices"]
    if not prices:
        last, vol, drift = 0.5, VOL_DEFAULT, 0.0
    else:
        last = prices[-1]
        if len(prices) < MIN_LEN:
            vol, drift = VOL_DEFAULT, 0.0
        else:
            vol = _volatility(prices)
            drift = _total_drift(prices, last)
    rows = []
    for h in range(1, horizon + 1):
        mid = last + drift * ((h / horizon) ** DRIFT_EXP)
        wide = vol * (h ** SPREAD_EXP)
        row = [_round6(mid + k * wide) for k in K]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for m in case["markets"]:
            forecasts.append({"id": m["id"], "q": market_rows(m, horizon)})
        out_path = out_dir / ("%s.out" % case_id)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
