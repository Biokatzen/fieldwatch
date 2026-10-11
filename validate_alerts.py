"""Blind validation of alerts: do alerted fields really look worse than non-alerted ones?

Workflow
  1) make : builds a contact sheet of Sentinel-2 true-colour thumbnails that MIXES alerted fields with
            random non-alerted control fields under anonymous ids (V01, V02, ...), plus an empty
            ratings.csv and a hidden key.csv. The sheet never shows which group a field belongs to.
  2) you  : open sheet.html, fill ratings.csv (verdict = problem | ok | unclear). Do not open key.csv.
  3) score: compares the problem rate in alerted vs control fields.

A "problem" is something you can see in the image: bare or patchy crop, failed/abandoned or already
tilled field next to green neighbours, etc. The Earth Engine/thumbnail part is NOT tested end to end;
the sampling and scoring logic is unit-tested.

Usage
  python validate_alerts.py make  --alerts reports/ks/alerts_2023-05-01.csv --fields data/ks/fields.gpkg \
         --week 2023-05-01 --project MY_PROJECT --out validation/ks_2023-05-01
  python validate_alerts.py score --ratings validation/ks_2023-05-01/ratings.csv \
         --key validation/ks_2023-05-01/key.csv
"""
from __future__ import annotations

import argparse
import base64
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

VERDICTS = {"problem", "ok", "unclear"}


# --------------------------------------------------------------------------- pure logic (tested)
def build_sheet_table(alerts: pd.DataFrame, all_field_ids, n_alerts: int = 12,
                      n_controls: int = 12, seed: int = 0) -> pd.DataFrame:
    """Alerted fields (most severe first) plus random controls, shuffled under anonymous ids."""
    rng = np.random.default_rng(seed)
    alerted = alerts.sort_values("severity", ascending=False).head(n_alerts)["field_id"].astype(str).tolist()
    pool = sorted(set(map(str, all_field_ids)) - set(alerts["field_id"].astype(str)))
    controls = [str(x) for x in rng.choice(pool, size=min(n_controls, len(pool)), replace=False)]
    df = pd.DataFrame([(f, "alert") for f in alerted] + [(f, "control") for f in controls],
                      columns=["field_id", "group"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["blind_id"] = [f"V{i + 1:02d}" for i in range(len(df))]
    return df


def score_ratings(ratings: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Problem rate per group; 'unclear' fields are reported but excluded from the rate."""
    d = key.merge(ratings[["blind_id", "verdict"]], on="blind_id", how="left")
    d["verdict"] = d["verdict"].astype("string").str.strip().str.lower()
    bad = d[~d["verdict"].isin(VERDICTS)]
    if len(bad):
        raise ValueError(f"unrated or invalid verdicts (use {sorted(VERDICTS)}): {list(bad['blind_id'])}")
    rows = []
    for group, sub in d.groupby("group"):
        rated = sub[sub["verdict"] != "unclear"]
        n_problem = int((rated["verdict"] == "problem").sum())
        rows.append({"group": group, "fields": len(sub), "unclear": len(sub) - len(rated),
                     "problems": n_problem, "rated": len(rated),
                     "problem_rate": n_problem / len(rated) if len(rated) else float("nan")})
    return pd.DataFrame(rows).set_index("group")


# --------------------------------------------------------------------------- Earth Engine (untested)
def _thumbnail_png(ee, geom: dict, week: str, size: int = 420) -> tuple[bytes, str]:
    """True-colour Sentinel-2 thumbnail around one field, outlined in yellow. Returns (png, image date)."""
    field = ee.Geometry(geom)
    region = field.buffer(250).bounds()
    w = ee.Date(week)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(field)
           .filterDate(w.advance(-7, "day"), w.advance(7, "day")).sort("CLOUDY_PIXEL_PERCENTAGE"))
    if col.size().getInfo() == 0:
        raise RuntimeError("no Sentinel-2 image within +-7 days")
    img = ee.Image(col.first())
    date = img.date().format("YYYY-MM-dd").getInfo()
    rgb = img.select(["B4", "B3", "B2"]).visualize(min=0, max=3000)
    outline = ee.FeatureCollection([ee.Feature(field)]).style(color="FFFF00", fillColor="00000000", width=2)
    url = rgb.blend(outline).getThumbURL({"region": region, "dimensions": size, "format": "png"})
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read(), date


def _write_sheet(path: Path, cards: list[dict], week: str) -> None:
    items = "".join(
        f'<div class="c"><b>{c["blind_id"]}</b> <span>{c["date"]}</span><br>'
        + (f'<img src="data:image/png;base64,{c["b64"]}">' if c["b64"] else f'<i>{c["error"]}</i>') + "</div>"
        for c in cards)
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Blind check {week}</title>
<style>body{{font-family:system-ui;margin:1.5rem}}.g{{display:flex;flex-wrap:wrap;gap:14px}}
.c{{width:420px}}.c span{{color:#666;font-size:12px}}img{{width:100%;border:1px solid #ccc}}</style></head><body>
<h2>Blind check, week of {week}</h2>
<p>Yellow outline = the field. Compare it with its neighbours and fill <code>ratings.csv</code>:
<b>problem</b> (visibly weaker/bare/failed than neighbours), <b>ok</b>, or <b>unclear</b> (clouds, can't tell).
Rate what you SEE only.</p><div class="g">{items}</div></body></html>""", encoding="utf-8")


def cmd_make(a) -> None:
    import ee
    import geopandas as gpd
    from shapely.geometry import mapping

    alerts = pd.read_csv(a.alerts)
    if alerts.empty:
        raise SystemExit("No alerts in that file")
    fields = gpd.read_file(a.fields).to_crs(4326).set_index("field_id")
    table = build_sheet_table(alerts, fields.index, a.n_alerts, a.n_controls, a.seed)
    ee.Initialize(project=a.project)
    cards = []
    for _, r in table.iterrows():
        try:
            png, date = _thumbnail_png(ee, mapping(fields.loc[r["field_id"], "geometry"]), a.week)
            cards.append({"blind_id": r["blind_id"], "date": date, "b64": base64.b64encode(png).decode()})
        except Exception as e:  # keep going; one failed thumbnail should not stop the sheet
            cards.append({"blind_id": r["blind_id"], "date": "", "b64": "", "error": f"no image ({e})"})
        print(f"{r['blind_id']}: {'ok' if cards[-1]['b64'] else cards[-1]['error']}")
    a.out.mkdir(parents=True, exist_ok=True)
    _write_sheet(a.out / "sheet.html", cards, a.week)
    table[["blind_id", "field_id", "group"]].to_csv(a.out / "key.csv", index=False)
    pd.DataFrame({"blind_id": table["blind_id"], "verdict": "", "notes": ""}).to_csv(a.out / "ratings.csv", index=False)
    print(f"\nOpen {a.out / 'sheet.html'}, fill {a.out / 'ratings.csv'}, and do NOT open key.csv until you finish.")


def cmd_score(a) -> None:
    res = score_ratings(pd.read_csv(a.ratings), pd.read_csv(a.key))
    print(res.round(2).to_string())
    if {"alert", "control"} <= set(res.index):
        print(f"\nproblem rate: alerted {res.loc['alert', 'problem_rate']:.0%} vs control {res.loc['control', 'problem_rate']:.0%}")
        print("(small samples: report the counts, not just the percentages)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--alerts", type=Path, required=True)
    m.add_argument("--fields", type=Path, required=True)
    m.add_argument("--week", required=True)
    m.add_argument("--project", required=True)
    m.add_argument("--out", type=Path, default=Path("validation/check"))
    m.add_argument("--n-alerts", type=int, default=12)
    m.add_argument("--n-controls", type=int, default=12)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(fn=cmd_make)
    s = sub.add_parser("score")
    s.add_argument("--ratings", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
