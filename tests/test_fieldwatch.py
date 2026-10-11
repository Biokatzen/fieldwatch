import numpy as np
import pandas as pd
import pytest

from fieldwatch import (AlertConfig, flag_fields, peer_zscores, season_mask, weekly_matrix)
from fetch_sentinel2_gee import rows_to_frame
from prepare_fields_csb import features_to_gdf
from run_weekly_report import build, parse_windows, to_html

CFG = AlertConfig()
WEEKS = pd.date_range("2025-04-07", periods=12, freq="7D")  # inside the winter-wheat window


def make_ts(n_fields=20, bad=None, dip_week=None, seed=0):
    """Healthy fields hover around 0.70; `bad` stays at 0.40 from week 4; `dip_week` has a one-week dip."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_fields):
        fid = f"F{i:02d}"
        for k, d in enumerate(WEEKS):
            v = 0.70 + rng.normal(0, 0.02)
            if fid == bad and k >= 4:
                v = 0.40 + rng.normal(0, 0.02)
            if fid == "F01" and k == dip_week:
                v = 0.35
            rows.append((fid, d + pd.Timedelta(days=2), v))
    ts = pd.DataFrame(rows, columns=["field_id", "date", "ndvi"])
    crops = pd.Series("winter_wheat", index=[f"F{i:02d}" for i in range(n_fields)])
    return ts, crops


def run(ts, crops, cfg=CFG):
    wide = weekly_matrix(ts, cfg)
    z = peer_zscores(wide, crops, cfg)
    mask = season_mask(wide.index, crops, wide.columns, cfg)
    return wide, z, flag_fields(z, mask, cfg)


def test_persistent_underperformer_is_flagged_and_healthy_fields_are_not():
    ts, crops = make_ts(bad="F05")
    _, _, flags = run(ts, crops)
    assert flags["F05"].iloc[-1]
    assert flags.drop(columns="F05").to_numpy().sum() == 0


def test_one_week_dip_does_not_alert():
    ts, crops = make_ts(dip_week=6)
    _, _, flags = run(ts, crops)
    assert not flags["F01"].any()


def test_results_never_use_future_data():
    ts, crops = make_ts(bad="F05")
    cutoff = WEEKS[7]
    _, z_full, f_full = run(ts, crops)
    ts_changed = ts.copy()
    future = pd.to_datetime(ts_changed["date"]) > cutoff + pd.Timedelta(days=6)
    ts_changed.loc[future, "ndvi"] = 0.05  # destroy everything after the cutoff
    _, z_cut, f_cut = run(ts_changed, crops)
    upto = z_full.index <= cutoff
    pd.testing.assert_frame_equal(z_full[upto], z_cut[upto])
    pd.testing.assert_frame_equal(f_full[upto], f_cut[upto])


def test_single_hazy_observation_after_cloud_gap_is_not_trusted():
    ts, crops = make_ts()
    gap = pd.to_datetime(ts["date"]).between(WEEKS[3], WEEKS[5] + pd.Timedelta(days=6)) & (ts["field_id"] == "F02")
    ts = ts[~gap | (pd.to_datetime(ts["date"]) == WEEKS[5] + pd.Timedelta(days=2))].copy()
    ts.loc[(ts["field_id"] == "F02") & (pd.to_datetime(ts["date"]) == WEEKS[5] + pd.Timedelta(days=2)), "ndvi"] = 0.3
    wide, _, flags = run(ts, crops)
    assert not flags["F02"].any()
    assert np.isnan(wide.loc[WEEKS[5], "F02"])  # reported as "no data", not as a fake low value


def test_no_alerts_outside_the_monitoring_window():
    ts, crops = make_ts(bad="F05")
    ts["date"] = pd.to_datetime(ts["date"]) + pd.Timedelta(days=150)  # shift to autumn: wheat window closed
    _, _, flags = run(ts, crops)
    assert flags.to_numpy().sum() == 0


def test_small_crop_groups_are_skipped_with_warning():
    ts, crops = make_ts(n_fields=5)
    wide = weekly_matrix(ts, CFG)
    with pytest.warns(UserWarning, match="not monitored"):
        z = peer_zscores(wide, crops, CFG)
    assert z.isna().all().all()


def test_report_builds_and_names_the_problem_field():
    ts, crops = make_ts(bad="F05")
    fields = crops.rename("crop").reset_index().rename(columns={"index": "field_id"})
    res = build(fields, ts, week=WEEKS[-1] + pd.Timedelta(days=1))
    assert list(res["alerts"]["field_id"]) == ["F05"]
    html = to_html(res, fields)
    assert "F05" in html and "<table" in html


def test_parse_windows_and_override_removes_late_season_alerts():
    assert parse_windows(["winter_wheat=04-01:05-31"]) == {"winter_wheat": ((4, 1), (5, 31))}
    assert parse_windows(None) == {}
    ts, crops = make_ts(bad="F05")  # problem visible in weeks up to ~Jun 23
    fields = crops.rename("crop").reset_index().rename(columns={"index": "field_id"})
    late = WEEKS[-1] + pd.Timedelta(days=1)
    assert len(build(fields, ts, week=late)["alerts"]) == 1
    short = AlertConfig()
    short.windows.update(parse_windows(["winter_wheat=04-01:05-01"]))  # window closed before `late`
    assert len(build(fields, ts, week=late, cfg=short)["alerts"]) == 0


def test_csb_features_are_converted_to_clean_fields():
    square = {"type": "Polygon", "coordinates": [[[-97.7, 38.8], [-97.69, 38.8], [-97.69, 38.81], [-97.7, 38.81], [-97.7, 38.8]]]}
    props = {"CSBID": "201623001649004", "CNTY": "Saline", "ACRES": 83.456, "CDL2023": 24}
    feats = [{"type": "Feature", "geometry": square, "properties": props},
             {"type": "Feature", "geometry": square, "properties": props}]  # duplicate id
    gdf = features_to_gdf(feats)
    assert list(gdf["field_id"]) == ["201623001649004"]
    assert gdf.loc[0, "crop"] == "winter_wheat" and gdf.loc[0, "acres"] == 83.5
    assert gdf.crs.to_epsg() == 4326


def test_empty_earth_engine_result_gives_clear_message_not_keyerror():
    with pytest.raises(SystemExit, match="No NDVI observations"):
        rows_to_frame([])


def test_rows_to_frame_cleans_and_sorts():
    rows = [{"field_id": "B", "date": "2023-04-02", "ndvi": 0.51234567},
            {"field_id": "A", "date": "2023-04-02", "ndvi": None},
            {"field_id": "A", "date": "2023-04-07", "ndvi": 0.6}]
    ts = rows_to_frame(rows)
    assert list(ts["field_id"]) == ["A", "B"] and ts.loc[1, "ndvi"] == 0.5123
    with pytest.raises(SystemExit, match="without columns"):
        rows_to_frame([{"field_id": "A", "date": "2023-04-02", "mean": 0.5}])  # the original bug's symptom
