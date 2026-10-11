# fieldwatch: weekly field alerts from satellite NDVI

[![tests](https://github.com/Biokatzen/fieldwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Biokatzen/fieldwatch/actions)

**Problem.** A seed or crop company monitors hundreds or thousands of fields, but agronomists can only
visit a few each week. Which ones should they visit first?

**Solution.** Every week, `fieldwatch` compares each field with its *peers* (same crop, same week) using
Sentinel-2 NDVI and flags fields that stay well below them for several weeks. It needs **no yield
labels**, and the output is a ranked list plus a one-page report that can be sent as-is.

![Weekly report, Kansas winter wheat, week of 2023-05-01](docs/kansas_report.png)

*Weekly report on real data (Kansas winter wheat).*

## How it works
1. **Weekly composite:** max-value composite per week (suppresses residual clouds), then a trailing
   median that needs at least 2 real observations. Gaps are reported as "no data", never invented.
2. **Peer comparison:** robust z-score against the median and MAD of the same crop that week. This
   removes the normal seasonal curve and weather effects.
3. **Persistence rule:** alert if the mean z over the last 3 weeks is <= -2. One cloudy week does not
   trigger an alert. These settings were chosen in advance and not tuned on the Kansas data.
4. **Monitoring window per crop**, so harvest and early-season differences don't create alerts.
5. **Report:** ranked table (severity, weeks flagged, last NDVI, pattern), field-vs-peers charts, map.

Everything is **causal**: week *t* uses only data up to week *t*, and a unit test enforces it.

## Results on real data: winter wheat, Kansas, 2023
150 winter-wheat fields (USDA Crop Sequence Boundaries, Saline County), Sentinel-2 NDVI from Earth Engine,
monitoring window 1 Apr - 31 May.

| Week | Alerts (out of 150 fields) |
|---|---|
| 2023-04-17 | 1 |
| 2023-05-01 | 9 |
| 2023-05-15 | 7 |
| 2023-05-29 | 5 |

Alerts are a prioritised list of fields worth inspecting, **not** a diagnosis.

**Validation status:** the blind check described below has not been completed yet, so there is
**no measured hit rate or false-alarm rate** for this method. Until then, treat the alerts as candidates
to inspect, not as confirmed problems.

### Blind validation of the alerts
Alert counts alone don't prove the method works. `validate_alerts.py` builds a sheet of Sentinel-2
thumbnails that **mixes alerted fields with random non-alerted controls under anonymous ids**, so the
ratings are not biased by knowing which is which. You then rate each field (problem / ok / unclear)
and the script compares the problem rate in alerted vs control fields.

### Regional or field-specific?
For the week of 2023-05-01, each alerted field was compared with its 8 nearest fields in the sample
(`neighborhood_check.py`, radius about 4-14 km). All 9 alerts were field-specific: they sit at least one
robust standard deviation below their neighbours, whose mean scores range from -0.64 to +0.25. Across all
150 fields, the correlation between a field's score and its neighbours' mean score was 0.01. So the alerts
are not explained by county-scale low-NDVI regions. This does not show that they are agronomic problems,
and because the 150 fields are a sample of the county it cannot see differences between immediately
adjacent fields.

## Limitations
- **No measured accuracy yet** (see validation status above). One county and one season only.
- Peers are defined by crop only. A field sown later than its neighbours can look like an underperformer;
  adding sowing date, variety or region to the peer definition is the natural next step.
- The monitoring window (1 Apr - 31 May) was set from typical Kansas harvest timing. Check the NASS weekly
  *Crop Progress* reports for your season so harvested fields are not read as failures.
- CSB polygons are algorithmic field estimates (not property boundaries) and crop labels come from
  satellite classification, not farm records.
- Detection is delayed by design: the 3-week persistence rule trades speed for fewer false alarms.
- Fields without enough clear observations are reported as "no data" rather than guessed.

## Use it
```bash
pip install -r requirements.txt
earthengine authenticate          # needs a Google Cloud project registered for Earth Engine

# 1) pick winter-wheat fields from USDA CSB (Kansas, Saline County, 2023)
python prepare_fields_csb.py --project MY_PROJECT --counties Saline --year 2023 --n 150 --out data/ks
# 2) download Sentinel-2 NDVI per field
python fetch_sentinel2_gee.py --fields data/ks/fields.gpkg --start 2023-03-01 --end 2023-06-15 \
       --project MY_PROJECT --out data/ks
# 3) weekly alert reports (CSV + HTML)
for w in 2023-04-17 2023-05-01 2023-05-15 2023-05-29; do
  python run_weekly_report.py --fields data/ks/fields.csv --ndvi data/ks/ndvi_timeseries.csv \
         --window winter_wheat=04-01:05-31 --week $w --out reports/ks
done
# 4) blind validation
python validate_alerts.py make --alerts reports/ks/alerts_2023-05-01.csv --fields data/ks/fields.gpkg \
       --week 2023-05-01 --project MY_PROJECT --out validation/ks_2023-05-01
#    open sheet.html, fill ratings.csv (problem / ok / unclear), then:
python validate_alerts.py score --ratings validation/ks_2023-05-01/ratings.csv \
       --key validation/ks_2023-05-01/key.csv

pytest                            # 15 unit tests
```
With your own fields, provide two CSVs: `fields.csv` (`field_id, crop[, lon, lat]`) and
`ndvi_timeseries.csv` (`field_id, date, ndvi`, cloud-masked).

## Files
| File | Purpose |
|---|---|
| `fieldwatch.py` | core logic: weekly composite, peer z-scores, alert rules |
| `run_weekly_report.py` | weekly CSV + HTML report |
| `prepare_fields_csb.py` | select winter-wheat fields from USDA Crop Sequence Boundaries |
| `fetch_sentinel2_gee.py` | Sentinel-2 NDVI download per field (Earth Engine) |
| `validate_alerts.py` | blind check: alerted vs control fields on Sentinel-2 thumbnails |
| `neighborhood_check.py` | tests whether alerts are field-specific or regional (compares with nearest fields) |
| `tests/` | 15 unit tests on small hand-made tables (they check the code logic, e.g. that no future data is used; they are not results) |

MIT License
