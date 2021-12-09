""" Script to convert ESO results to OCF ML format """
from pathlib import Path

import pandas as pd
import xarray as xr
from nowcasting_dataset.data_sources.gsp import eso
from nowcasting_dataset.geospatial import lat_lon_to_osgb
from tqdm import tqdm

# # Convert ESO results

# Idea is to convert to raw ESO results into the same format than the outputs from OCF ML models

# ## Load ESO file

# Load the ESO file

# ## Load ESO GSP metadata

# Load the ESO GSP metadata. We will need to convert between GSP ID and GSP name,
# and know the location of each GSP

# ## Load test dataset

# Let load the test dataset meta file. This is so we know which times
# and what locations / gsps to compare

# ## Reduce ESO forecasts

# Reduce ESO forecast to the same as the test dataset

# ## Load Forecast Truth and Capacity

# Load the GSP outturn value at each forecast value. Also load the capacity

# ************

# location fo ESO forecasts
ESO_PV_FORECASTS_PATH = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/other_organisations_pv_forecasts/"
    "National_Grid_ESO/NetCDF/ESO_GSP_PV_forecasts.nc"
)

# The locations of the tests dataset
TEST_DATASET_FILE = (
    "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/"
    "nowcasting_dataset_pipeline/prepared_ML_training_data/"
    "v16/test/spatial_and_temporal_locations_of_each_example.csv"
)

# The "ground truth" estimated total PV generation from each Grid Supply Point from Sheffield Solar:
GSP_ZARR_PATH = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/"
    "nowcasting_dataset_pipeline/PV/GSP/v3/pv_gsp.zarr"
)

# Output csv
ESO_PV_FORECASTS_OUTPUT_FILE = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/other_organisations_pv_forecasts/"
    "National_Grid_ESO/CSV/testset_v16.csv"
)

# ************
print("Load main file")

eso_pv_forecasts_dataset = xr.open_dataset(ESO_PV_FORECASTS_PATH)
eso_pv_forecasts_dataset = eso_pv_forecasts_dataset.rename({"gsp_id": "gsp_name"})
eso_pv_forecasts_dataset

# ************
print("ESO 30 mins to 2 hours")

# Select just two timesteps: 30 minutes to 2 hours ahead:
ESO_FORECAST_ALGO_NAME = "ASL"  # Either ASL or ML.
selected_eso_forecasts_dataarray = eso_pv_forecasts_dataset[ESO_FORECAST_ALGO_NAME].sel(
    step=slice(pd.Timedelta("30 minutes"), pd.Timedelta("2 hour"))
)

selected_eso_forecasts_dataarray

# ************
print("Load ESO file")
print("Load ESO GSP metadata")

# Download metadata from ESO for each GSP.  We need this because
# the GSP PV data from Sheffield Solar uses integer gsp_ids, whilst
# the ESO forecasts use textual gsp_names.  So we need the metadata
# to allow us to map from gsp_id to gsp_name.
gsp_metadata = eso.get_gsp_metadata_from_eso(calculate_centroid=True)

gsp_id_to_name = (
    gsp_metadata[["gsp_id", "gsp_name"]]
    .dropna()
    .astype({"gsp_id": int})
    .set_index("gsp_id")
    .squeeze()
)

gsp_name_to_id = (
    gsp_metadata[["gsp_id", "gsp_name"]]
    .dropna()
    .astype({"gsp_name": str})
    .set_index("gsp_name")
    .squeeze()
)
print(gsp_id_to_name)

gsp_metadata["location_x"], gsp_metadata["location_y"] = lat_lon_to_osgb(
    lat=gsp_metadata["centroid_lat"], lon=gsp_metadata["centroid_lon"]
)

print(gsp_metadata.columns)
print(gsp_metadata[["location_x", "location_y"]])

# ************
print("Load test dataset")
# get list of datetimes fof test set

# load location for test dataset
locations_df = pd.read_csv(TEST_DATASET_FILE)

# append gsp id to locations_df


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

assert locations_df["gsp_id"].isna().sum() == 0

# ************
print("Reduce ESO forecasts")

# only select the 'forecast_date_time' from test set

# change them to have mins of 0 or 30, as ESO forecasts are at 0 or 30 mins.
locations_df["t0_datetime_UTC_floor_30_mins"] = pd.to_datetime(
    locations_df["t0_datetime_UTC"]
).dt.floor("30T")
forecast_date_time = selected_eso_forecasts_dataarray.forecast_date_time

# make list of datetimes that are in 't0_datetime_utc' and 'forecast_date_time'
t0_datetime_utc = pd.DatetimeIndex(locations_df["t0_datetime_UTC_floor_30_mins"]).unique()
forecast_date_time = pd.DatetimeIndex(forecast_date_time)
forecast_date_time = forecast_date_time.join(t0_datetime_utc, how="inner")

# select the eso forecasts that are in the test set

print(forecast_date_time.max())
print(forecast_date_time.min())
selected_eso_forecasts_dataarray = selected_eso_forecasts_dataarray.sel(
    forecast_date_time=forecast_date_time
)

# also filter t0_datetimes
locations_df = locations_df[locations_df["t0_datetime_UTC_floor_30_mins"].isin(forecast_date_time)]
print(locations_df)

# Dont want to load any unnecessary data, so going to loop through t0_datetime_UTC and only save the forecast we need

eso_dataarrays_list = []
# this can take about 2 seconds for ~5000 samples
for i in tqdm(range(len(locations_df))):
    t0_datetime_UTC_floor_30_mins = locations_df.t0_datetime_UTC_floor_30_mins.iloc[i]
    gsp_id = locations_df.gsp_id.iloc[i]
    gsp_name = gsp_id_to_name[gsp_id]

    one_eso_forecasts_dataarray = selected_eso_forecasts_dataarray.sel(
        forecast_date_time=t0_datetime_UTC_floor_30_mins
    )
    one_eso_forecasts_dataarray = one_eso_forecasts_dataarray.sel(gsp_name=gsp_name)

    eso_dataarrays_list.append(one_eso_forecasts_dataarray)

predictions = []
# this can take about 20 seconds for ~5000 samples
for eso_dataarrays in tqdm(eso_dataarrays_list):
    target_times = eso_dataarrays.forecast_date_time + eso_dataarrays.step
    forecast = eso_dataarrays.values
    gsp_id = gsp_name_to_id[eso_dataarrays.gsp_name.values]

    predictions_df = pd.DataFrame(
        {"forecast_gsp_pv_outturn_mw": forecast, "target_datetime_utc": target_times}
    )
    predictions_df["gsp_id"] = int(gsp_id)
    predictions_df["t0_datetime_utc"] = eso_dataarrays.forecast_date_time.values

    predictions.append(predictions_df)

predictions = pd.concat(predictions)
print(predictions)

# ************
print("Load Forecast Truth and Capacity")

# now we need to find out the truth values
pv_live_dataset = xr.open_dataset(GSP_ZARR_PATH, engine="zarr")

# Convert 'gsp_id' from strings like '1', '2', etc. to ints
pv_live_dataset["gsp_id"] = pv_live_dataset["gsp_id"].astype(int)

predictions_and_truths = predictions
predictions_and_truths["actual_gsp_pv_outturn_mw"] = 0.0
predictions_and_truths["capacity_mwp"] = 0.0
# this can take about 1 min 30 seconds for ~5000 samples
for i in tqdm(range(len(predictions_and_truths))):
    target_datetime_utc = predictions_and_truths.target_datetime_utc.iloc[i]
    gsp_id = predictions_and_truths.gsp_id.iloc[i]

    one_pv_live_dataset = pv_live_dataset.sel(datetime_gmt=target_datetime_utc)
    one_pv_live_dataset = one_pv_live_dataset.sel(gsp_id=gsp_id)

    predictions_and_truths.actual_gsp_pv_outturn_mw.iloc[
        i
    ] = one_pv_live_dataset.generation_mw.values
    predictions_and_truths.capacity_mwp.iloc[i] = one_pv_live_dataset.installedcapacity_mwp.values

print(predictions_and_truths)

# is genration mw normalized?


# ************
print("Save file to csv")
# save file to csv
predictions_and_truths.to_csv(ESO_PV_FORECASTS_OUTPUT_FILE)
