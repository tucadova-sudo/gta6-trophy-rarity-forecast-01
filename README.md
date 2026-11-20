# GTA VI Trophy Rarity Forecaster

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast
each PlayStation 5 trophy's Exophase earn rate at 2026-11-26 00:00 UTC, as
five quantiles per trophy (5%, 25%, 50%, 75%, 95%).

## Program

`main.py` (Python 3.13, standard library only) reads every `<id>.in` case
in the input directory and writes `<id>.out` forecasts:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Method

Two forces move a trophy's earn rate: players earn trophies as they
progress (rates rise) while new players keep joining the tracked pool
(rates dilute). The forecast is an ensemble of four per-trophy models on
the logit scale:

- **M0 persistence** — the rate stays where the last read put it.
- **M1 log-time trend** — least squares of logit(rate) on ln(1 + hours
  since the first read), read at the target.
- **M2 dilution** — least squares of logit(rate) on ln(players tracked),
  with players tracked at the target extrapolated from a log-time fit,
  capped at ~7x the last read's count.
- **M3 two-factor** — least squares on both ln(1 + hours) and
  ln(players tracked) together.

The models are weighted by a rolling-origin backtest run inside the case's
own history (train before an origin, score the median against the read six
hours later, weight by inverse error). The spread combines a random-walk
term (0.5 logit per root-day) with the backtest residual scale, widened
with the horizon, so the band stays honest over a multi-day horizon with
several Exophase refreshes in it.

Output is deterministic: two runs on the same case write the same bytes,
well inside the 8-second limit (about 0.1s on the practice cases).

## License

MIT — see LICENSE.
