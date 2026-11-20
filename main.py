"""GTA VI trophy rarity forecaster for gta6-trophy-rarity-forecast-01.

Reads every <id>.in case in the input directory and writes an <id>.out
forecast: for each of the game's trophies, five quantiles (5%, 25%, 50%,
75%, 95%) of its Exophase earn rate at the case's target.

Method (standard library only, deterministic, no network):
- Each case's history is a series of reads of Exophase's trophy page:
  (time, players tracked, earn rate per trophy). Two forces move a rate:
  players earn trophies as they progress (rates rise) while new players
  keep joining and diluting the tracked pool (rates fall). So the median
  forecast is an ensemble of four per-trophy models:
    M0 persistence: the rate stays where the last read put it.
    M1 log-time trend: least squares of logit(rate) on ln(1 + hours since
       the first read), read at the target.
    M2 dilution: least squares of logit(rate) on ln(players tracked),
       with players tracked at the target extrapolated from its own
       log-linear fit against time.
    M3 two-factor: least squares of logit(rate) on both ln(1 + hours)
       and ln(players tracked) together.
- The models are weighted by a rolling-origin backtest run inside the
  case's own history: train each model on reads before an origin, score
  its median against the read six hours later, and weight by inverse
  error. The final median is the weighted mean of the models' logits.
- Spread: a random-walk term (0.5 logit per root-day, as the house
  baseline uses) combined with the backtest residual scale, widened with
  the horizon, so the band stays honest over a multi-day horizon with
  several Exophase refreshes in it.
- Output: one row per trophy, quantiles from the median and spread on the
  logit scale, rounded to 4 decimals, clamped to [0.01, 99.99] and
  non-decreasing. Two runs on the same case write the same bytes.

The grader runs `python3 -P -s /bot/main.py` in python:3.13-slim with no
network and nothing installed, so this file is self-contained.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Backtest horizon in hours: how far ahead each rolling origin predicts.
BACKTEST_HOURS = 6.0
# Trend models' deviation from the last rate is damped by this much: a
# trend fitted over days should not be trusted at full strength 4.5 days out.
TREND_DAMP = 0.8


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def ols1(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_std, n)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    if n == 1:
        return (ys[0], 0.0, 0.0, 1)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return (a, b, resid, n)


def ols2(xs1, xs2, ys):
    """Least squares y ~ a + b1*x1 + b2*x2 via normal equations.

    Returns (a, b1, b2, resid_std, n). Falls back to the one-factor fits
    when the system is singular.
    """
    n = len(ys)
    if n < 4:
        a1, b1, _, _ = ols1(xs1, ys)
        return (a1, b1, 0.0, 0.0, n)
    m1 = sum(xs1) / n
    m2 = sum(xs2) / n
    my = sum(ys) / n
    c11 = sum((x - m1) ** 2 for x in xs1)
    c22 = sum((x - m2) ** 2 for x in xs2)
    c12 = sum((x1 - m1) * (x2 - m2) for x1, x2 in zip(xs1, xs2))
    d1 = sum((x1 - m1) * (y - my) for x1, y in zip(xs1, ys))
    d2 = sum((x2 - m2) * (y - my) for x2, y in zip(xs2, ys))
    det = c11 * c22 - c12 * c12
    if abs(det) <= 1e-12 * max(1.0, c11 * c22):
        a1, b1, _, _ = ols1(xs1, ys)
        return (a1, b1, 0.0, 0.0, n)
    b1 = (d1 * c22 - d2 * c12) / det
    b2 = (d2 * c11 - d1 * c12) / det
    a = my - b1 * m1 - b2 * m2
    if n > 3:
        var = sum((y - (a + b1 * x1 + b2 * x2)) ** 2
                  for x1, x2, y in zip(xs1, xs2, ys)) / (n - 3)
        resid = math.sqrt(max(0.0, var))
    else:
        resid = 0.0
    return (a, b1, b2, resid, n)


class TrophySeries:
    """One trophy's reads: (hours since first read, ln players, logit rate)."""

    __slots__ = ("hours", "ln_n", "lg", "last_lg")

    def __init__(self):
        self.hours = []
        self.ln_n = []
        self.lg = []
        self.last_lg = None


def build_series(case):
    """Turn a case's history into per-trophy series plus read-level data."""
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    t0 = parse_time(history[0]["at"])
    reads = []  # (hours, ln(players_tracked))
    series = [TrophySeries() for _ in case["trophies"]]
    for read in history:
        t = parse_time(read["at"])
        hours = (t - t0) / 3600.0
        players = read.get("players_tracked") or 0
        ln_n = math.log(players) if players > 0 else 0.0
        reads.append((hours, ln_n))
        rates = read["rates"]
        for i, s in enumerate(series):
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            s.hours.append(hours)
            s.ln_n.append(ln_n)
            s.lg.append(logit(r))
    for s in series:
        if s.lg:
            s.last_lg = s.lg[-1]
    return reads, series


def model_predictions(series, reads, target_hours, target_ln_n):
    """The four models' median logits for one trophy at the target.

    Returns [M0, M1, M2, M3] as logits. Models that cannot be fitted fall
    back to persistence.
    """
    last = series.last_lg if series.last_lg is not None else logit(1.0)
    if len(series.lg) < 3:
        return [last, last, last, last]
    x_target = math.log1p(max(0.0, target_hours))
    xs = [math.log1p(max(0.0, h)) for h in series.hours]
    a1, b1, _, _ = ols1(xs, series.lg)
    m1 = a1 + b1 * x_target
    a2, b2, _, _ = ols1(series.ln_n, series.lg)
    m2 = a2 + b2 * target_ln_n
    a3, b3, b4, _, _ = ols2(xs, series.ln_n, series.lg)
    m3 = a3 + b3 * x_target + b4 * target_ln_n
    # Damp the trend models' move away from the last rate.
    m1 = last + (m1 - last) * TREND_DAMP
    m2 = last + (m2 - last) * TREND_DAMP
    m3 = last + (m3 - last) * TREND_DAMP
    return [last, m1, m2, m3]


def backtest_weights(case, reads, series_list):
    """Rolling-origin backtest: weight each model by inverse median error.

    Origins are reads with at least 6 hours of history behind them and a
    read at least BACKTEST_HOURS ahead to score against. Returns
    (weights, resid_scale): weights sum to 1, resid_scale is the
    backtest's mean absolute logit error across models, for the spread.
    """
    default = [0.4, 0.2, 0.2, 0.2]
    if len(reads) < 4:
        return default, 0.5
    # Candidate origins: every read from the 4th on, spaced >= 2h apart.
    origins = []
    last_origin_h = None
    for j in range(3, len(reads)):
        h = reads[j][0]
        if last_origin_h is not None and h - last_origin_h < 2.0:
            continue
        # Find the first read at least BACKTEST_HOURS after this origin.
        k = None
        for kk in range(j + 1, len(reads)):
            if reads[kk][0] - h >= BACKTEST_HOURS:
                k = kk
                break
        if k is None:
            continue
        origins.append((j, k))
        last_origin_h = h
        if len(origins) >= 6:
            break
    if not origins:
        return default, 0.5
    errors = [0.0, 0.0, 0.0, 0.0]
    counts = [0, 0, 0, 0]
    for j, k in origins:
        target_h = reads[k][0]
        target_ln_n = reads[k][1]
        for i, s in enumerate(series_list):
            # Train on reads up to origin j only.
            sub = TrophySeries()
            for h, ln_n, lg in zip(s.hours, s.ln_n, s.lg):
                if h <= reads[j][0] + 1e-9:
                    sub.hours.append(h)
                    sub.ln_n.append(ln_n)
                    sub.lg.append(lg)
            if sub.lg:
                sub.last_lg = sub.lg[-1]
            # Truth: the rate at read k, from the full series.
            truth = None
            for h, lg in zip(s.hours, s.lg):
                if abs(h - target_h) < 1e-9:
                    truth = lg
                    break
            if truth is None:
                continue
            preds = model_predictions(sub, reads, target_h, target_ln_n)
            for m in range(4):
                errors[m] += abs(preds[m] - truth)
                counts[m] += 1
    if not counts[0]:
        return default, 0.5
    mean_err = [errors[m] / counts[m] if counts[m] else 1.0
                for m in range(4)]
    raw = [1.0 / (e + 0.05) for e in mean_err]
    total = sum(raw)
    weights = [r / total for r in raw]
    resid = sum(mean_err) / 4.0
    return weights, resid


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("case asks for levels %s, not %s"
                         % (case["levels"], LEVELS))
    reads, series_list = build_series(case)
    last_h, last_ln_n = reads[-1]
    horizon_days = case["horizon_minutes"] / 1440.0
    target_h = last_h + case["horizon_minutes"] / 60.0
    # Extrapolate players tracked to the target: log-linear in log-time,
    # which stays conservative when growth slows (a linear-in-time fit
    # would explode), and capped at ~7x the last read's count.
    if len(reads) >= 2:
        xs = [math.log1p(max(0.0, h)) for h, _ in reads]
        ys = [ln for _, ln in reads]
        a_n, b_n, _, _ = ols1(xs, ys)
        target_ln_n = a_n + b_n * math.log1p(max(0.0, target_h))
        target_ln_n = min(target_ln_n, last_ln_n + 2.0)
        # Never extrapolate the player count backwards.
        target_ln_n = max(target_ln_n, last_ln_n)
    else:
        target_ln_n = last_ln_n
    weights, resid = backtest_weights(case, reads, series_list)
    # Fallback median for a trophy with no rates anywhere: the
    # cross-trophy median of last logits.
    known = sorted(s.last_lg for s in series_list if s.last_lg is not None)
    fallback_lg = known[len(known) // 2] if known else logit(1.0)
    # Spread: random-walk term plus backtest residual, widened with horizon.
    rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
    back_h = BACKTEST_HOURS / 24.0
    scaled = resid * 1.25 * math.sqrt(max(horizon_days, 0.0) / back_h)
    spread = math.sqrt(rw * rw + scaled * scaled)
    spread = min(2.5, max(0.35, spread))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        s = series_list[i]
        if s.last_lg is None:
            median = fallback_lg
        else:
            preds = model_predictions(s, reads, target_h, target_ln_n)
            median = sum(w * p for w, p in zip(weights, preds))
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp(inv_logit(median + z * spread)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
