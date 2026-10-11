"""Generate a realistic synthetic Sentinel-2 NDVI dataset with known, injected problems.

Realism included: crop-specific phenology (double-logistic curve), per-field sowing-date
variation, regional weather shifts, cloud gaps that hit whole regions at once, and residual
undetected clouds that create false low-NDVI points. A share of fields get a real problem
(sudden drop, gradual stress, poor emergence) at a known date, so detection can be scored.

Usage:  python make_demo_data.py            # writes data/demo/*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fieldwatch import SEASON_WINDOWS

# day-of-year parameters of the NDVI curve per crop
CROPS = {
    "winter_wheat": dict(n=50, up=80, down=165, s_up=9, s_down=10, peak=(0.78, 0.90), base=(0.15, 0.25)),
    "maize": dict(n=40, up=160, down=255, s_up=12, s_down=10, peak=(0.75, 0.88), base=(0.12, 0.22)),
    "sugar_beet": dict(n=30, up=130, down=285, s_up=14, s_down=12, peak=(0.78, 0.90), base=(0.12, 0.22)),
}
YEAR = 2025
ANOMALY_TYPES = ["sudden_drop", "gradual_stress", "poor_emergence"]


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _doy(month_day) -> int:
    return pd.Timestamp(YEAR, *month_day).dayofyear


def make_demo(seed: int = 7, anomaly_rate: float = 0.14):
    rng = np.random.default_rng(seed)
    doys = np.arange(60, 274, 5)  # Mar 1 .. Sep 30, 5-day revisit
    n_regions = 4
    region_shift = rng.normal(0, 4, n_regions)               # weather: days of phenology shift
    region_cloud = rng.random((len(doys), n_regions)) < 0.35  # clouds cover a whole region on a date
    region_xy = rng.uniform([8, 47], [12, 52], (n_regions, 2))

    fields, series, truth = [], [], []
    fid = 0
    for crop, p in CROPS.items():
        w_start, w_end = (_doy(d) for d in SEASON_WINDOWS[crop])
        for _ in range(p["n"]):
            fid += 1
            field_id = f"F{fid:03d}"
            region = int(rng.integers(n_regions))
            shift = region_shift[region] + rng.normal(0, 5)   # sowing-date variation
            base, peak = rng.uniform(*p["base"]), rng.uniform(*p["peak"])
            clean = base + (peak - base) * (_sig((doys - (p["up"] + shift)) / p["s_up"])
                                            - _sig((doys - (p["down"] + 0.5 * shift)) / p["s_down"]))
            mult = np.ones_like(clean)

            if rng.random() < anomaly_rate:
                kind = str(rng.choice(ANOMALY_TYPES))
                if kind == "sudden_drop":
                    onset = int(rng.integers(w_start + 21, w_end - 28)); f = rng.uniform(0.45, 0.70)
                    mult = np.where(doys >= onset, f, 1.0)
                elif kind == "gradual_stress":
                    onset = int(rng.integers(w_start + 21, w_end - 49)); f = rng.uniform(0.55, 0.75)
                    mult = 1 - (1 - f) * np.clip((doys - onset) / 42, 0, 1)
                else:  # poor_emergence: weak stand from the start of the season
                    onset = int(max(p["up"] + shift - 10, w_start)); f = rng.uniform(0.60, 0.80)
                    mult = np.where(doys >= onset, f, 1.0)
                truth.append({"field_id": field_id, "anomaly": kind, "severity_factor": round(f, 2),
                              "onset_date": (pd.Timestamp(YEAR, 1, 1) + pd.Timedelta(days=onset - 1)).date()})

            ndvi = base + (clean - base) * mult + rng.normal(0, 0.02, len(doys))
            seen = ~region_cloud[:, region] & (rng.random(len(doys)) > 0.10)
            haze = rng.random(len(doys)) < 0.05                 # undetected cloud: spurious low value
            ndvi = np.where(haze, ndvi * rng.uniform(0.4, 0.8, len(doys)), ndvi)
            dates = pd.Timestamp(YEAR, 1, 1) + pd.to_timedelta(doys - 1, unit="D")
            series.append(pd.DataFrame({"field_id": field_id, "date": dates[seen], "ndvi": ndvi[seen].round(4)}))
            xy = region_xy[region] + rng.normal(0, 0.12, 2)
            fields.append({"field_id": field_id, "crop": crop, "region": region,
                           "lon": round(xy[0], 4), "lat": round(xy[1], 4)})

    truth_df = pd.DataFrame(truth, columns=["field_id", "anomaly", "severity_factor", "onset_date"])
    return pd.DataFrame(fields), pd.concat(series, ignore_index=True), truth_df


if __name__ == "__main__":
    out = Path("data/demo")
    out.mkdir(parents=True, exist_ok=True)
    f, ts, t = make_demo()
    f.to_csv(out / "fields.csv", index=False)
    ts.to_csv(out / "ndvi_timeseries.csv", index=False)
    t.to_csv(out / "anomaly_truth.csv", index=False)
    print(f"{len(f)} fields, {len(ts)} observations, {len(t)} injected problems -> {out}/")
