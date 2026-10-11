"""fieldwatch: weekly field-monitoring alerts from satellite NDVI time series.

Idea: a field is suspicious when it stays far below its *peers* (same crop, same week)
for several weeks in a row. Comparing with peers removes the normal seasonal curve
and weather effects, so no yield labels are needed.

Everything here is causal: results for week t only use data up to week t, so the
backtest in evaluate.py reflects what would happen in real operation.

Input tables
------------
fields: field_id, crop            (optional: lon, lat)
ndvi:   field_id, date, ndvi      (cloud-masked Sentinel-2 NDVI, one row per observation)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Monitoring window per crop ((month, day) start, end). Outside it no alerts are raised:
# early and late in the season fields differ legitimately (sowing date, harvest).
SEASON_WINDOWS = {
    "winter_wheat": ((4, 1), (6, 30)),
    "maize": ((6, 1), (9, 10)),
    "sugar_beet": ((5, 15), (9, 30)),
}


@dataclass
class AlertConfig:
    z_thresh: float = 2.0      # alert when mean robust z over the last `persist` weeks <= -z_thresh
    persist: int = 3           # weeks of persistence (suppresses one-off cloud artefacts)
    smooth_weeks: int = 3      # trailing median window (weeks) applied to the weekly composite
    smooth_min_obs: int = 2    # real weekly observations required in that window (else 'no data')
    min_peers: int = 8         # smallest crop group that can be monitored
    min_scale: float = 0.05    # floor for the robust spread (NDVI units), avoids huge z in uniform weeks
    windows: dict = field(default_factory=lambda: dict(SEASON_WINDOWS))


# --------------------------------------------------------------------------- preprocessing
def weekly_matrix(ts: pd.DataFrame, cfg: AlertConfig, start=None, end=None) -> pd.DataFrame:
    """Weeks x fields NDVI matrix.

    1. Max-value composite per ISO week (standard way to suppress residual clouds)
    2. Trailing 'higher' median over ``smooth_weeks`` (causal: no future data), requiring at least
       ``smooth_min_obs`` real observations so one hazy pixel after a cloud gap cannot
       masquerade as a persistent drop. Otherwise the week stays NaN ('no data').
    """
    ts = ts.dropna(subset=["ndvi"]).copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts["week"] = ts["date"].dt.to_period("W").dt.start_time
    wide = ts.pivot_table(index="week", columns="field_id", values="ndvi", aggfunc="max")
    start = pd.Timestamp(start) if start is not None else wide.index.min()
    end = pd.Timestamp(end) if end is not None else wide.index.max()
    wide = wide.reindex(pd.date_range(start, end, freq="7D"))
    # 'higher' median: with two observations take the larger one. Clouds only ever LOWER NDVI,
    # so this is the conservative choice (the cost is up to one extra week to confirm a real drop).
    return wide.rolling(cfg.smooth_weeks, min_periods=cfg.smooth_min_obs).quantile(0.5, interpolation="higher")


def season_mask(index: pd.DatetimeIndex, crops: pd.Series, columns, cfg: AlertConfig) -> pd.DataFrame:
    """Boolean weeks x fields matrix: True inside the crop's monitoring window."""
    mask = pd.DataFrame(False, index=index, columns=list(columns))
    for crop, ((sm, sd), (em, ed)) in cfg.windows.items():
        cols = [c for c in crops[crops == crop].index if c in mask.columns]
        if not cols:
            continue
        inside = np.array([(pd.Timestamp(d.year, sm, sd) <= d <= pd.Timestamp(d.year, em, ed)) for d in index])
        mask.loc[inside, cols] = True
    return mask


# --------------------------------------------------------------------------- peer comparison
def peer_stats(wide: pd.DataFrame, cols: list[str], cfg: AlertConfig) -> tuple[np.ndarray, np.ndarray]:
    """Weekly peer median and robust spread (1.4826 * MAD, floored) for one crop group."""
    X = wide[cols].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN weeks
        med = np.nanmedian(X, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(X - med), axis=1, keepdims=True) * 1.4826
    return med, np.maximum(mad, cfg.min_scale)


def peer_zscores(wide: pd.DataFrame, crops: pd.Series, cfg: AlertConfig) -> pd.DataFrame:
    """Robust z-score of each field against the other fields of the same crop, week by week."""
    z = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    for crop, ids in crops.groupby(crops).groups.items():
        cols = [c for c in ids if c in wide.columns]
        if len(cols) < cfg.min_peers:
            warnings.warn(f"crop '{crop}' has only {len(cols)} fields (< {cfg.min_peers}); not monitored")
            continue
        med, scale = peer_stats(wide, cols, cfg)
        z[cols] = (wide[cols].to_numpy() - med) / scale
    return z


# --------------------------------------------------------------------------- alerts
def persistent_score(z: pd.DataFrame, cfg: AlertConfig) -> pd.DataFrame:
    """Mean z over the trailing ``persist`` weeks (needs at least 2 valid weeks)."""
    return z.rolling(cfg.persist, min_periods=2).mean()


def flag_fields(z: pd.DataFrame, mask: pd.DataFrame, cfg: AlertConfig) -> pd.DataFrame:
    """Boolean weeks x fields matrix of alerts."""
    return (persistent_score(z, cfg) <= -cfg.z_thresh) & mask


def flag_streaks(flags: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive weeks each field has been flagged up to each week."""
    f = flags.to_numpy()
    out = np.zeros(f.shape, dtype=int)
    for i in range(len(f)):
        out[i] = (out[i - 1] + 1) * f[i] if i else f[i].astype(int)
    return pd.DataFrame(out, index=flags.index, columns=flags.columns)


def alert_table(wide, z, flags, crops, week, cfg: AlertConfig) -> pd.DataFrame:
    """Alerts for one week, most severe first."""
    week = pd.Timestamp(week)
    score = persistent_score(z, cfg).loc[week]
    streak = flag_streaks(flags).loc[week]
    drop3 = (z - z.shift(3)).loc[week]
    ids = flags.columns[flags.loc[week].to_numpy()]
    rows = []
    for fid in ids:
        crop = crops[fid]
        recent = wide[fid].loc[:week].iloc[-cfg.smooth_weeks:].dropna()
        age = int((week - recent.index[-1]).days // 7) if len(recent) else None
        med, _ = peer_stats(wide, [c for c in crops[crops == crop].index if c in wide.columns], cfg)
        rows.append({
            "field_id": fid,
            "crop": crop,
            "severity": round(-score[fid], 2),
            "weeks_flagged": int(streak[fid]),
            "last_ndvi": round(float(recent.iloc[-1]), 3) if len(recent) else np.nan,
            "weeks_since_obs": age,
            "peer_median_ndvi": round(float(med[wide.index.get_loc(week), 0]), 3),
            "pattern": "sudden drop" if drop3[fid] <= -3 else "persistent underperformance",
        })
    cols = ["field_id", "crop", "severity", "weeks_flagged", "last_ndvi", "weeks_since_obs",
            "peer_median_ndvi", "pattern"]
    return pd.DataFrame(rows, columns=cols).sort_values("severity", ascending=False, ignore_index=True)
