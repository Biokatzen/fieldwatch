"""Download per-field Sentinel-2 NDVI time series with Google Earth Engine (GEE).

!! The Earth Engine calls are NOT tested end to end by the author (no access to the live API).
!! A first version silently returned 0 rows (property name bug); fixed with setOutputs.
!! Expect to adjust small things (auth, project name, field-count limits). The output format
!! is what matters: the two CSVs below are all the rest of the project needs.

Input : field boundaries (GeoJSON / GeoPackage / Shapefile) with columns `field_id` and `crop`
        (crop = one of winter_wheat, maize, sugar_beet, or edit SEASON_WINDOWS in fieldwatch.py).
        Your company's own parcel records are ideal. Open alternatives: EuroCrops and national
        parcel registers (check which countries/years are available).
Output: <out>/fields.csv           field_id, crop, lon, lat
        <out>/ndvi_timeseries.csv  field_id, date, ndvi

Setup:  pip install earthengine-api geopandas ; earthengine authenticate
Usage:  python fetch_sentinel2_gee.py --fields my_fields.gpkg --start 2025-03-01 --end 2025-09-30 \
            --project my-gcp-project --out data/real
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

VALID_SCL = [4, 5]  # Sentinel-2 scene classification: 4 = vegetation, 5 = bare soil (drops clouds, shadow, snow, water)


def _prep(img):
    """Mask clouds with the SCL band and add an NDVI band."""
    scl = img.select("SCL")
    good = scl.eq(VALID_SCL[0]).Or(scl.eq(VALID_SCL[1]))
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    return ndvi.updateMask(good).copyProperties(img, ["system:time_start"])


def _chunk(fc, start: str, end: str) -> tuple[int, list[dict]]:
    """Return (number of Sentinel-2 images found, per-field NDVI rows) for one date range."""
    import ee

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(fc.geometry()).filterDate(start, end))
    n_images = col.size().getInfo()
    if n_images == 0:
        return 0, []
    col = col.map(_prep)
    # setOutputs pins the property name to "ndvi". Without it, reduceRegions on a single-band
    # image names it after the reducer ("mean"), and every row is silently filtered out.
    reducer = ee.Reducer.mean().setOutputs(["ndvi"])

    def per_image(img):
        date = img.date().format("YYYY-MM-dd")
        reduced = img.reduceRegions(collection=fc, reducer=reducer, scale=10, tileScale=4)
        return reduced.map(lambda f: f.set("date", date))

    out = col.map(per_image).flatten().filter(ee.Filter.notNull(["ndvi"]))
    return n_images, [f["properties"] for f in out.getInfo()["features"]]


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows from Earth Engine -> tidy table, with a clear message instead of a cryptic KeyError."""
    if not rows:
        raise SystemExit(
            "No NDVI observations were returned for any date range.\n"
            "Check the per-chunk lines above: '0 images' means the dates/area are wrong; "
            "'N images, 0 observations' means every polygon was fully masked (clouds) or the "
            "property name is not 'ndvi'. Try --inner-buffer-m 0 and a date range in the growing season."
        )
    ts = pd.DataFrame(rows)
    missing = {"field_id", "date", "ndvi"} - set(ts.columns)
    if missing:
        raise SystemExit(f"Earth Engine returned rows without columns {sorted(missing)}; got {list(ts.columns)}")
    ts = ts[["field_id", "date", "ndvi"]].dropna()
    ts["ndvi"] = ts["ndvi"].round(4)
    return ts.sort_values(["field_id", "date"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--out", type=Path, default=Path("data/real"))
    ap.add_argument("--inner-buffer-m", type=float, default=10, help="shrink polygons to avoid mixed edge pixels")
    ap.add_argument("--chunk-days", type=int, default=14, help="smaller chunks if you hit GEE size limits")
    a = ap.parse_args()

    import ee

    ee.Initialize(project=a.project)
    gdf = gpd.read_file(a.fields)[["field_id", "crop", "geometry"]].to_crs(4326)
    metric = gdf.estimate_utm_crs()
    gdf["geometry"] = gdf.to_crs(metric).buffer(-a.inner_buffer_m).to_crs(4326)
    gdf = gdf[~gdf.geometry.is_empty]
    fc = ee.FeatureCollection(json.loads(gdf[["field_id", "geometry"]].to_json()))

    a.out.mkdir(parents=True, exist_ok=True)
    cent = gdf.to_crs(metric).geometry.centroid.to_crs(4326)
    pd.DataFrame({"field_id": gdf["field_id"], "crop": gdf["crop"], "lon": cent.x.round(5), "lat": cent.y.round(5)}
                 ).to_csv(a.out / "fields.csv", index=False)

    rows = []
    edges = pd.date_range(a.start, a.end, freq=f"{a.chunk_days}D").append(pd.DatetimeIndex([a.end])).unique()
    for s, e in zip(edges[:-1], edges[1:]):
        n_images, new = _chunk(fc, s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
        rows += new
        print(f"{s.date()} -> {e.date()}: {n_images} images, {len(new)} observations ({len(rows)} total)")
    ts = rows_to_frame(rows)
    ts.to_csv(a.out / "ndvi_timeseries.csv", index=False)
    print(f"done: {len(ts)} observations for {ts['field_id'].nunique()} fields -> {a.out}/")


if __name__ == "__main__":
    main()
