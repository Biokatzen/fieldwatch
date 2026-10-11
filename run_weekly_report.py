"""Create the weekly alert report (CSV + self-contained HTML) from NDVI time series.

Usage:
  python run_weekly_report.py --fields data/demo/fields.csv --ndvi data/demo/ndvi_timeseries.csv \
         --week 2025-06-16 --out reports
Without --week the latest week in the data is used.
"""
from __future__ import annotations

import argparse
import base64
import io
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fieldwatch import (AlertConfig, alert_table, flag_fields, peer_stats, peer_zscores,
                        season_mask, weekly_matrix)

CSS = """body{font-family:system-ui,Arial,sans-serif;max-width:980px;margin:2rem auto;color:#222;padding:0 1rem}
h1{margin-bottom:0}.sub{color:#666;margin-top:.2rem}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:6px 10px;border-bottom:1px solid #ddd;text-align:left}th{background:#f4f6f4}
.kpi{display:flex;gap:1rem;margin:1rem 0}.kpi div{background:#f4f6f4;padding:.8rem 1.2rem;border-radius:8px}
.kpi b{font-size:1.6rem;display:block}img{max-width:100%}.note{color:#666;font-size:13px}"""


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_fields(wide, crops, alerts, week, cfg, n=6):
    top = alerts.head(n)
    if top.empty:
        return None
    ncols = 3
    nrows = -(-len(top) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.9 * nrows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        cols = [c for c in crops[crops == r["crop"]].index if c in wide.columns]
        med, scale = peer_stats(wide, cols, cfg)
        idx = wide.index[wide.index <= week]
        m, s = med[: len(idx), 0], scale[: len(idx), 0]
        ax.fill_between(idx, m - 2 * s, m + 2 * s, color="#bbb", alpha=0.5, label="peers ±2 spread")
        ax.plot(idx, m, color="#777", lw=1, label="peer median")
        ax.plot(idx, wide.loc[idx, r["field_id"]], color="#c0392b", lw=2, label="this field")
        ax.set_title(f"{r['field_id']}  ({r['crop']}, severity {r['severity']})", fontsize=9)
        ax.tick_params(labelsize=7); ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=30)
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    return _png(fig)


def plot_map(fields, alerts):
    if not {"lon", "lat"} <= set(fields.columns):
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(fields["lon"], fields["lat"], s=14, color="#cfd8cf", label="monitored fields")
    if not alerts.empty:
        a = alerts.merge(fields[["field_id", "lon", "lat"]], on="field_id")
        sc = ax.scatter(a["lon"], a["lat"], c=a["severity"], cmap="YlOrRd", s=60, edgecolor="k", linewidth=0.5)
        fig.colorbar(sc, label="severity")
    ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def parse_windows(items: list[str] | None) -> dict:
    """['winter_wheat=04-01:05-31'] -> {'winter_wheat': ((4, 1), (5, 31))}"""
    out = {}
    for item in items or []:
        crop, span = item.split("=")
        start, end = (tuple(int(x) for x in part.split("-")) for part in span.split(":"))
        out[crop] = (start, end)
    return out


def build(fields: pd.DataFrame, ts: pd.DataFrame, week=None, cfg: AlertConfig | None = None):
    cfg = cfg or AlertConfig()
    crops = fields.set_index("field_id")["crop"]
    wide = weekly_matrix(ts, cfg)
    week = pd.Timestamp(week).to_period("W").start_time if week else wide.index.max()
    wide = wide.loc[wide.index <= week]  # strictly no data after the report week
    z = peer_zscores(wide, crops, cfg)
    mask = season_mask(wide.index, crops, wide.columns, cfg)
    flags = flag_fields(z, mask, cfg)
    alerts = alert_table(wide, z, flags, crops, week, cfg)
    monitored = int(mask.loc[week].sum())
    no_data = int((mask.loc[week] & wide.loc[week].isna()).sum())
    return dict(week=week, wide=wide, crops=crops, alerts=alerts, monitored=monitored,
                no_data=no_data, cfg=cfg)


def to_html(res: dict, fields: pd.DataFrame) -> str:
    week, alerts, cfg = res["week"], res["alerts"], res["cfg"]
    fig1 = plot_fields(res["wide"], res["crops"], alerts, week, cfg)
    fig2 = plot_map(fields, alerts)
    img = lambda b: f'<img src="data:image/png;base64,{b}">' if b else ""
    table = alerts.to_html(index=False, border=0) if not alerts.empty else "<p>No alerts this week.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Field alerts {week.date()}</title>
<style>{CSS}</style></head><body>
<h1>Weekly field alerts</h1><p class="sub">Week of {week.date()}  |  Sentinel-2 NDVI, peer comparison by crop</p>
<div class="kpi"><div><b>{res['monitored']}</b>fields in season</div><div><b>{len(alerts)}</b>alerts</div>
<div><b>{res['no_data']}</b>without recent data</div></div>
<h2>Fields to visit first</h2>{table}
<p class="note">Severity = how many robust standard deviations the field sits below its crop peers, averaged over
the last {cfg.persist} weeks. Alert when severity &ge; {cfg.z_thresh}. A late-sown field can also appear here:
check sowing date before sending an agronomist.</p>
<h2>Most severe fields vs. their peers</h2>{img(fig1)}
<h2>Where they are</h2>{img(fig2)}
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--ndvi", type=Path, required=True)
    ap.add_argument("--week", default=None, help="any date inside the report week (default: latest)")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--window", action="append", metavar="CROP=MM-DD:MM-DD",
                    help="override a crop's monitoring window, e.g. winter_wheat=04-01:05-31 (repeatable)")
    a = ap.parse_args()
    cfg = AlertConfig()
    cfg.windows.update(parse_windows(a.window))

    fields, ts = pd.read_csv(a.fields), pd.read_csv(a.ndvi, parse_dates=["date"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = build(fields, ts, a.week, cfg)
    for x in w:
        print("warning:", x.message)
    a.out.mkdir(parents=True, exist_ok=True)
    tag = res["week"].date()
    res["alerts"].to_csv(a.out / f"alerts_{tag}.csv", index=False)
    (a.out / f"report_{tag}.html").write_text(to_html(res, fields), encoding="utf-8")
    print(f"week {tag}: {len(res['alerts'])} alerts among {res['monitored']} fields in season -> {a.out}/")


if __name__ == "__main__":
    main()
