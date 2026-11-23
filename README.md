# Virtual Power Plant Composition and Day-Ahead Solar Forecasting

## The task

A platform operator holds about 250 small solar plants and must split them into two aggregated resources (virtual power plants) and forecast each resource's hourly output one day ahead. The score is the settlement S: revenue minus a penalty proportional to the forecast error and the hourly electricity price (SMP). Only information published before 17:00 KST of the previous day may be used.

## Approach

1. **Data audit and panel** (`01_data_checks.py`, `01b_review_checks.py`, `02_build_panel.py`): a site-hour panel of the providers' day-ahead forecasts, calendar, neighbour-station and physics (clear-sky) features.
2. **Hold-out experiments** (`03_forecast_probes.py`, `03c_blend.py`): time-split hold-out years scored with the competition metric. Every setting was fixed on hold-out windows, never on the leaderboard.
3. **External weather forecasts** (`13_`–`18_`, `20_nwp_cells.py`, `26_gfs_extra_national.py`, `gfs_features.py`, `regional_features.py`): NOAA GFS / GEFS / GEFS-Aerosols and ECMWF IFS runs of the previous day, reduced to national and regional summaries that are identical for every plant.
4. **Models** (`08_make_submission.py`, `nn_member.py`, `gru_member.py`): LightGBM and XGBoost members, multi-layer perceptrons and a day-sequence GRU, trained with an SMP-weighted L1 loss and combined by a family blend that lowers the network share on days when the two families disagree (`blend_utils.py`, `21_reblend.py`).
5. **Composition** (`05_composition.py`): splits that keep sister plants apart so that their errors cancel inside each resource.
6. **Checks** (`09_check_submission.py`, `10_test_causality.py`, `29_pubtime_rule.py`): format checks, a leakage test (destroying all later days' forecasts must not change any feature), and a scan of the posting time of 48,426 weather-forecast files so that no file posted after the 17:00 cut-off is used.

## Running

The competition data is **not included** (the rules allow it to be used only for the contest). Put the participant dataset in a folder and point `OIBC_DATA` at it; intermediate files go to `OIBC_CACHE` (default `analysis_code/_cache`). The external weather tables are downloaded by the `13_`–`20_`, `26_`–`28_` scripts into `analysis_code/external_data/`.

```bash
pip install -r analysis_code/requirements_main.txt      # requirements_grib.txt for the GRIB downloads
export OIBC_DATA=/path/to/participant_dataset
cd analysis_code
python 02_build_panel.py
python 08_make_submission.py --recipe v6 --help
```
