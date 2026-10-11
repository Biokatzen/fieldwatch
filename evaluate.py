"""Backtest: how many real problems does fieldwatch catch, and how many false alarms does it cost?

Compared on identical data (several random seeds), all strictly causal:
  fieldwatch  robust peer z-score + 3-week persistence          (this project)
  peer_abs    NDVI more than X below the crop median, 3 weeks    (simple peer rule)
  fixed       NDVI below a fixed value for 3 weeks               (what many dashboards do)

Metrics
  false alarm rate = share of healthy field-weeks (inside the monitoring window) that are flagged
  recall           = share of problem fields flagged at some week on/after the problem onset
  delay            = weeks between problem onset and the first alert
Methods are compared at EQUAL false alarm rate, which is the fair way to compare detectors.

Usage:  python evaluate.py
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fieldwatch import (AlertConfig, flag_fields, peer_stats, peer_zscores, season_mask, weekly_matrix)
from make_demo_data import make_demo

SEEDS = range(100, 110)  # held-out seeds: not used while designing/debugging the method
START, END = "2025-03-03", "2025-09-29"
DEFAULT_CFG = AlertConfig()


def _prepare(seed: int):
    fields, ts, truth = make_demo(seed)
    crops = fields.set_index("field_id")["crop"]
    wide = weekly_matrix(ts, DEFAULT_CFG, START, END)
    mask = season_mask(wide.index, crops, wide.columns, DEFAULT_CFG)
    return wide, crops, mask, truth


def _rolling_flag(stat: pd.DataFrame, mask, thr, how: str, persist: int = 3) -> pd.DataFrame:
    r = stat.rolling(persist, min_periods=2)
    if how == "mean_below":
        return (r.mean() <= -thr) & mask
    return (r.max() < thr) & mask  # "all below" for the fixed rule


def _peer_diff(wide, crops, cfg) -> pd.DataFrame:
    d = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    for _, ids in crops.groupby(crops).groups.items():
        cols = [c for c in ids if c in wide.columns]
        med, _ = peer_stats(wide, cols, cfg)
        d[cols] = wide[cols].to_numpy() - med
    return d


def score(flags: pd.DataFrame, mask: pd.DataFrame, truth: pd.DataFrame) -> dict:
    bad = list(truth["field_id"])
    healthy = [c for c in flags.columns if c not in bad]
    far = flags[healthy].to_numpy().sum() / max(mask[healthy].to_numpy().sum(), 1)
    detected, delays = 0, []
    for _, t in truth.iterrows():
        onset_week = pd.Timestamp(t["onset_date"]).to_period("W").start_time
        after = flags.loc[flags.index >= onset_week, t["field_id"]]
        if after.any():
            detected += 1
            delays.append((after.idxmax() - onset_week).days / 7)
    return {"far": far, "detected": detected, "n_problems": len(bad), "delays": delays}


def run_all() -> pd.DataFrame:
    sweeps = {
        "fieldwatch": ("z threshold", np.arange(1.0, 5.01, 0.5)),
        "peer_abs": ("NDVI margin", np.arange(0.02, 0.201, 0.02)),
        "fixed": ("NDVI threshold", np.arange(0.20, 0.701, 0.05)),
    }
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in SEEDS:
            wide, crops, mask, truth = _prepare(seed)
            z = peer_zscores(wide, crops, DEFAULT_CFG)
            diff = _peer_diff(wide, crops, DEFAULT_CFG)
            for method, (_, grid) in sweeps.items():
                for thr in grid:
                    if method == "fieldwatch":
                        cfg = AlertConfig(z_thresh=float(thr))
                        flags = flag_fields(z, mask, cfg)
                    elif method == "peer_abs":
                        flags = _rolling_flag(diff, mask, thr, "mean_below")
                    else:
                        flags = _rolling_flag(wide, mask, thr, "all_below")
                    s = score(flags, mask, truth)
                    rows.append({"seed": seed, "method": method, "thr": round(float(thr), 3), **s})
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        delays = [d for ds in g["delays"] for d in ds]
        return pd.Series({
            "far": g["far"].mean(),
            "recall": g["detected"].sum() / g["n_problems"].sum(),
            "delay_weeks": float(np.median(delays)) if delays else np.nan,
            "n_problems": int(g["n_problems"].sum()),
        })
    return res.groupby(["method", "thr"]).apply(agg, include_groups=False).reset_index()


def recall_at_far(curve: pd.DataFrame, far_target: float) -> float:
    c = curve.sort_values("far")
    return float(np.interp(far_target, c["far"], c["recall"]))


def main() -> None:
    res = run_all()
    summ = summarise(res)
    summ.to_csv("docs/evaluation_curves.csv", index=False)

    ours = summ[(summ.method == "fieldwatch") & np.isclose(summ.thr, DEFAULT_CFG.z_thresh)].iloc[0]
    print(f"Backtest: {len(list(SEEDS))} simulated seasons, {int(ours.n_problems)} injected problems, 120 fields each")
    print(f"\nfieldwatch at default settings (z >= {DEFAULT_CFG.z_thresh}, {DEFAULT_CFG.persist} weeks):")
    print(f"  recall                 {ours.recall:.0%}")
    print(f"  median delay           {ours.delay_weeks:.1f} weeks after onset")
    print(f"  false alarm rate       {ours.far:.1%} of healthy field-weeks "
          f"(~{ours.far * 100:.1f} false alerts per 100 fields per week)")
    print("\nRecall at the SAME false alarm rate as fieldwatch default:")
    table = []
    for m in ["fieldwatch", "peer_abs", "fixed"]:
        r = recall_at_far(summ[summ.method == m], ours.far)
        print(f"  {m:11s} {r:.0%}")
        table.append({"method": m, "recall_at_equal_far": round(r, 3)})
    pd.DataFrame(table).to_csv("docs/evaluation_equal_far.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"fieldwatch": "#2e7d32", "peer_abs": "#1f77b4", "fixed": "#c0392b"}
    for m, c in colors.items():
        s = summ[summ.method == m].sort_values("far")
        ax[0].plot(s["far"] * 100, s["recall"] * 100, marker="o", ms=3, color=c, label=m)
    ax[0].scatter([ours.far * 100], [ours.recall * 100], s=90, facecolor="none", edgecolor="k", zorder=5, label="default setting")
    ax[0].set_xlim(0, 12); ax[0].set_xlabel("False alerts (% of healthy field-weeks)")
    ax[0].set_ylabel("Problem fields detected (%)"); ax[0].legend(frameon=False)
    ax[0].set_title("Detection vs. false alarms (higher-left is better)")

    d = []
    for seed in SEEDS:
        wide, crops, mask, truth = _prepare(seed)
        z = peer_zscores(wide, crops, DEFAULT_CFG)
        fl = flag_fields(z, mask, DEFAULT_CFG)
        for _, t in truth.iterrows():
            ow = pd.Timestamp(t["onset_date"]).to_period("W").start_time
            a = fl.loc[fl.index >= ow, t["field_id"]]
            d.append({"type": t["anomaly"], "delay": (a.idxmax() - ow).days / 7 if a.any() else np.nan})
    d = pd.DataFrame(d)
    by = d.groupby("type").agg(detected=("delay", lambda s: s.notna().mean()), delay=("delay", "median"))
    by.to_csv("docs/evaluation_by_problem_type.csv")
    print("\nBy problem type (fieldwatch default):")
    print(by.round(2).to_string())
    ax[1].bar(by.index, by["detected"] * 100, color="#2e7d32")
    for i, (t, r) in enumerate(by.iterrows()):
        ax[1].text(i, r["detected"] * 100 + 2, f"delay {r['delay']:.1f} wk", ha="center", fontsize=9)
    ax[1].set_ylim(0, 115); ax[1].set_ylabel("Detected (%)"); ax[1].set_title("Detection by problem type")
    plt.setp(ax[1].get_xticklabels(), rotation=15)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("docs/evaluation.png", dpi=150)


if __name__ == "__main__":
    main()
