"""Are the alerts field-specific, or is the whole neighbourhood low?

fieldwatch compares each field with ALL fields of the same crop. If a field is low only because it sits in a
low-NDVI area (later sowing, other soil or variety), its nearest neighbours will be low too. This script
compares each field's persistent score with the mean score of its k nearest fields:

    gap = own score - mean score of the k nearest fields      (negative = worse than its neighbours)

An alert is called "field-specific" if gap <= -GAP_THRESHOLD, otherwise "regional". The 1.0 threshold is an
arbitrary choice (one robust standard deviation), so look at the raw numbers too.

It also prints how strongly a field's score follows its neighbours' scores across ALL fields.

Usage:
  python neighborhood_check.py --fields data/ks/fields.csv --ndvi data/ks/ndvi_timeseries.csv \
         --window winter_wheat=04-01:05-31 --week 2023-05-01 --k 8
Needs lon/lat columns in fields.csv (fetch_sentinel2_gee.py writes them).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from fieldwatch import AlertConfig, peer_zscores, persistent_score
from run_weekly_report import build, parse_windows

GAP_THRESHOLD = 1.0


def neighbor_gap(score: pd.Series, coords: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Own score vs. mean score of the k nearest fields (distances in km from lon/lat)."""
    ids = score.dropna().index.intersection(coords.index)
    if len(ids) <= k:
        raise ValueError(f"need more than k={k} fields with a score, got {len(ids)}")
    lat0 = coords.loc[ids, "lat"].mean()
    xy = np.c_[coords.loc[ids, "lon"].to_numpy() * 111.32 * np.cos(np.radians(lat0)),
               coords.loc[ids, "lat"].to_numpy() * 110.57]
    dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    s = score.loc[ids].to_numpy()
    rows = []
    for i, fid in enumerate(ids):
        nn = np.argsort(dist[i])[:k]
        rows.append((fid, s[i], s[nn].mean(), dist[i, nn].max()))
    out = pd.DataFrame(rows, columns=["field_id", "score", "neighbor_mean", "radius_km"]).set_index("field_id")
    out["gap"] = out["score"] - out["neighbor_mean"]
    return out


def classify(table: pd.DataFrame, alert_ids, gap_threshold: float = GAP_THRESHOLD) -> pd.DataFrame:
    alerts = table.loc[table.index.intersection(list(alert_ids))].copy()
    alerts["type"] = np.where(alerts["gap"] <= -gap_threshold, "field-specific", "regional")
    return alerts.sort_values("score")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--ndvi", type=Path, required=True)
    ap.add_argument("--week", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--window", action="append", metavar="CROP=MM-DD:MM-DD")
    ap.add_argument("--out", type=Path, default=None, help="optional CSV with the alert table")
    a = ap.parse_args()

    fields = pd.read_csv(a.fields)
    ts = pd.read_csv(a.ndvi, parse_dates=["date"])
    cfg = AlertConfig()
    cfg.windows.update(parse_windows(a.window))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = build(fields, ts, a.week, cfg)
        z = peer_zscores(res["wide"], res["crops"], cfg)
    score = persistent_score(z, cfg).loc[res["week"]]
    table = neighbor_gap(score, fields.set_index("field_id")[["lon", "lat"]], a.k)

    r = np.corrcoef(table["score"], table["neighbor_mean"])[0, 1]
    print(f"week {res['week'].date()}, k={a.k} nearest fields")
    print(f"correlation between a field's score and its neighbours' mean score (all fields): {r:.2f}")
    print("  (close to 1: scores are mostly regional; close to 0: mostly field by field)\n")
    out = classify(table, res["alerts"]["field_id"])
    if out.empty:
        print("no alerted fields with a neighbourhood score this week")
        return
    print(out.round(2).to_string())
    n_spec = int((out["type"] == "field-specific").sum())
    print(f"\n{n_spec} of {len(out)} alerts are field-specific "
          f"(gap <= -{GAP_THRESHOLD:g}); {len(out) - n_spec} look regional")
    if a.out:
        out.to_csv(a.out)


if __name__ == "__main__":
    main()
