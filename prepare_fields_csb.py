"""Select winter-wheat fields from the USDA Crop Sequence Boundaries (CSB) in Google Earth Engine.

Column names below were taken from a real CSB feature (Saline County, Kansas):
CDL2023 (24 = winter wheat), CROP23, ACRES, CNTY, STATEFIPS, CSBID.
Ran successfully on real data (Kansas, 2023); the conversion to a GeoDataFrame is unit-tested.

CSB polygons are algorithmic field estimates (not property boundaries) and the crop label comes
from satellite classification. Say so when you present results.

Setup : pip install earthengine-api geopandas
Usage : python prepare_fields_csb.py --project MY_PROJECT --counties Saline --year 2023 --n 150 --out data/ks
Output: <out>/fields.gpkg  (field_id, crop, county, acres, geometry) -> input for fetch_sentinel2_gee.py
Tip   : keep all fields in one or a few neighbouring counties so peers share the same weather.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

CSB_ASSET = "projects/sat-io/open-datasets/USDA/CSB_1623"  # CSB covering 2016-2023
WINTER_WHEAT_CDL = 24
MIN_PEERS = 8


def features_to_gdf(features: list[dict]) -> gpd.GeoDataFrame:
    """GeoJSON features (as returned by Earth Engine getInfo) -> tidy GeoDataFrame."""
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    out = gpd.GeoDataFrame({
        "field_id": gdf["CSBID"].astype(str),
        "crop": "winter_wheat",
        "county": gdf["CNTY"],
        "acres": gdf["ACRES"].round(1),
    }, geometry=gdf.geometry, crs=4326)
    return out.drop_duplicates("field_id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--counties", nargs="+", required=True, help="county names as in CSB, e.g. Saline")
    ap.add_argument("--state-fips", default="20", help="20 = Kansas")
    ap.add_argument("--year", type=int, default=2023, help="harvest year, 2016-2023 for this CSB version")
    ap.add_argument("--min-acres", type=float, default=30, help="drop small polygons (edge pixels dominate)")
    ap.add_argument("--max-acres", type=float, default=500)
    ap.add_argument("--n", type=int, default=150, help="number of fields to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/ks"))
    a = ap.parse_args()
    if not 2016 <= a.year <= 2023:
        raise SystemExit("This CSB asset only covers 2016-2023")

    import ee  # imported here so the module can be imported (and tested) without earthengine-api

    ee.Initialize(project=a.project)
    query = ee.FeatureCollection(CSB_ASSET).filter(ee.Filter.And(
        ee.Filter.eq("STATEFIPS", a.state_fips),
        ee.Filter.inList("CNTY", a.counties),
        ee.Filter.eq(f"CDL{a.year}", WINTER_WHEAT_CDL),
        ee.Filter.gte("ACRES", a.min_acres),
        ee.Filter.lte("ACRES", a.max_acres),
    ))
    total = query.size().getInfo()
    print(f"{total} winter-wheat polygons between {a.min_acres:g} and {a.max_acres:g} acres in {a.counties}")
    if total < MIN_PEERS:
        raise SystemExit(f"Need at least {MIN_PEERS} fields; add counties or relax the acre limits")

    sample = query.randomColumn("r", a.seed).sort("r").limit(a.n)
    gdf = features_to_gdf(sample.getInfo()["features"])
    a.out.mkdir(parents=True, exist_ok=True)
    gdf.to_file(a.out / "fields.gpkg", driver="GPKG")
    print(f"saved {len(gdf)} fields (median {gdf['acres'].median():.0f} acres) -> {a.out / 'fields.gpkg'}")


if __name__ == "__main__":
    main()
