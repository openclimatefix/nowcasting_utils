"""
Baseline model for predicting GSP level results for by using yesterdays results
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import xarray as xr
from nowcasting_dataset.data_sources.gsp import eso
from nowcasting_dataset.geospatial import lat_lon_to_osgb
from tqdm import tqdm

from nowcasting_utils.metrics.evaluation import evaluation

# The locations of the tests dataset
TEST_DATASET_FILE = (
    "s3://solar-pv-nowcasting-data/prepared_ML_training_data/v16/"
    "test/spatial_and_temporal_locations_of_each_example.csv"
)

# The "ground truth" estimated total PV generation from each Grid Supply Point from Sheffield Solar:
GSP_ZARR_PATH = "gs://solar-pv-nowcasting-data/PV/GSP/v3/pv_gsp.zarr"

# Output csv
BASELINE_PV_FORECASTS_OUTPUT_FILE = Path("baseline_yesterday_testset_v16.csv")

# Load test set results
locations_df = pd.read_csv(TEST_DATASET_FILE)

# make list of datetimes that are in 't0_datetime_utc' and 'forecast_date_time'
locations_df["t0_datetime_UTC_floor_30_mins"] = pd.to_datetime(
    locations_df["t0_datetime_UTC"]
).dt.floor("30T")
locations_df = locations_df.groupby(["t0_datetime_UTC_floor_30_mins"]).first().reset_index()

# change center to gsp
gsp_metadata = eso.get_gsp_metadata_from_eso(calculate_centroid=True)

gsp_metadata["location_x"], gsp_metadata["location_y"] = lat_lon_to_osgb(
    lat=gsp_metadata["centroid_lat"], lon=gsp_metadata["centroid_lon"]
)
gsp_metadata_format = gsp_metadata[["location_x", "location_y", "gsp_id"]].rename(
    columns={"location_x": "x_center_OSGB", "location_y": "y_center_OSGB"}
)

# round coordinates so that merge works
gsp_metadata_format["x_center_OSGB"] = gsp_metadata_format["x_center_OSGB"].round(6)
gsp_metadata_format["y_center_OSGB"] = gsp_metadata_format["y_center_OSGB"].round(6)
locations_df["x_center_OSGB"] = locations_df["x_center_OSGB"].round(6)
locations_df["y_center_OSGB"] = locations_df["y_center_OSGB"].round(6)

locations_df = locations_df.merge(
    gsp_metadata_format, on=["x_center_OSGB", "y_center_OSGB"], how="left"
)


# load all gsp results
pv_live_dataset = xr.open_dataset(GSP_ZARR_PATH, engine="zarr")


predictions_and_truths = locations_df
predictions_and_truths["actual_gsp_pv_outturn_mw"] = -1.0
predictions_and_truths["forecast_gsp_pv_outturn_mw"] = -1.0
predictions_and_truths["capacity_mwp"] = -1.0
predictions_and_truths.t0_datetime_UTC = pd.to_datetime(
    predictions_and_truths.t0_datetime_UTC_floor_30_mins
)
# 'target_datetime_utc'

predictions_and_truths["t0_datetime_utc"] = predictions_and_truths.t0_datetime_UTC

results_df = []
forecast_horizons = [
    timedelta(hours=0.5),
    timedelta(hours=1),
    timedelta(hours=1.5),
    timedelta(hours=2),
]


for i in tqdm(range(len(predictions_and_truths))):
    # for i in tqdm(range(10)):
    t0_datetime_utc = predictions_and_truths.t0_datetime_UTC.iloc[i]
    yesterday_datetime_utc = t0_datetime_utc - timedelta(hours=24)

    target_datetimes_utc = [t0_datetime_utc + delta for delta in forecast_horizons]
    yesterday_datetimes_utc = [yesterday_datetime_utc + delta for delta in forecast_horizons]

    # yesterday
    one_pv_live_dataset = pv_live_dataset.sel(datetime_gmt=yesterday_datetime_utc)
    forecast_gsp_pv_outturn_mw = (
        one_pv_live_dataset.generation_mw.to_dataframe()
        .reset_index()
        .rename(columns={"datetime_gmt": "yesterday_datetime_utc"})
    )
    forecast_gsp_pv_outturn_mw["target_datetime_utc"] = forecast_gsp_pv_outturn_mw[
        "yesterday_datetime_utc"
    ] + timedelta(hours=24)

    # actual
    one_pv_live_dataset = pv_live_dataset.sel(datetime_gmt=target_datetimes_utc)
    actual_gsp_pv_outturn_mw = (
        one_pv_live_dataset.generation_mw.to_dataframe()
        .reset_index()
        .rename(columns={"datetime_gmt": "target_datetime_utc"})
    )
    capacity_mwp = (
        one_pv_live_dataset.installedcapacity_mwp.to_dataframe()
        .reset_index()
        .rename(columns={"datetime_gmt": "target_datetime_utc"})
    )

    results_df_one = forecast_gsp_pv_outturn_mw.rename(
        columns={"generation_mw": "forecast_gsp_pv_outturn_mw"}
    )
    results_df_one["actual_gsp_pv_outturn_mw"] = actual_gsp_pv_outturn_mw["generation_mw"]
    results_df_one["capacity_mwp"] = capacity_mwp["installedcapacity_mwp"]
    results_df_one["t0_datetime_utc"] = t0_datetime_utc

    results_df.append(results_df_one)


results_df_all = pd.concat(results_df)


# save csv
print(results_df_all)
results_df_all.to_csv(BASELINE_PV_FORECASTS_OUTPUT_FILE)


# run evaluation
evaluation(results_df=results_df_all, model_name="yesterday")
