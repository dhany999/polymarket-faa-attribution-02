# polymarket-faa-attribution-02

Forecast the Yes price of every Polymarket FAA outage-attribution market,
minute by minute, 4 hours ahead — five quantiles (0.05 / 0.25 / 0.5 / 0.75 /
0.95) per minute.

## Model

Jump-aware random walk with a slow pull home, per market:

```
q(tau, h) = clamp(mid(h) + k(tau) * vol * h**0.5, 0, 1)
```

- **vol** — per-market spread from the size of recent 1-minute moves:
  mean absolute 1-minute change, each move clipped at 0.02 so one huge
  jump cannot inflate the spread, averaged over a short (120 min) and a
  long (720 min) window, blended 50/50, converted mean-absolute to
  standard-deviation units (×1.2533), floored at 0.0002, capped at 0.02.
- **mid(h)** — the checkpoint price drifting linearly over the horizon
  toward the market's recent (480 min) median price: 25% of the gap,
  capped at ±0.03.
- **k(tau)** — heavy-tailed multipliers `[-2.5, -0.70, 0, 0.70, 2.5]`,
  wider in the tails than a normal distribution because jumps happen.
- Markets with fewer than 10 prices (the final's FAA markets start with
  short histories) fall back to the house baseline spread 0.0025 with no
  drift; a market with no prices at all is centred at 0.5. The program
  never crashes on an odd case.

Parameters were tuned on the six practice cases (NVIDIA CEO event,
same shape as the final): pooled pinball loss on the five locally
scorable cases p00–p04 is **0.304029**, versus the house baseline at
0.392518 (−22.5%) and the untuned winsorized-sigma starting point at
0.317564 (−19.1%). Per case: p00 0.4615 (base 0.5361), p01 0.2779
(0.3804), p02 0.3595 (0.4084), p03 0.2639 (0.3686), p04 0.1945
(0.2988). Tuning was one-factor-at-a-time over ~70 variants in four
rounds; the winning moves over the starting point were a shorter
volatility window (120 min), a 50/50 short/long blend, a wider winsor
clip (0.02), a lower vol floor (0.0002) and wider tail multipliers
(±2.5). Each helped independently and stacked monotonically.

## Run

```sh
python3 main.py
```

Reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).
Python 3.13, standard library only, no network, no randomness,
deterministic. One case runs in ~0.2 s, far under the 8 s cap.

## Files

- `main.py` — the entrypoint
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet claim
