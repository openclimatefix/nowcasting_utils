#!/usr/bin/env python
# coding: utf-8

"""
This script converts the multiple CSV files from ESO into a single NetCDF file.

The output NetCDF file has two DataArrays: 'ASL' and 'ML'.

ESO uses two PV forecasting algorithms in the Platform for Energy Forecasting (PEF):

* ASL: "Advanced Statistical Learning": These are hand-crafted models written in `R` by the
       ESO forecasting team.
* ML: "Maching Learning": ML models, trained from historical data, created by the ESO Labs team.

Each DataArray has three dimensions:

1. `gsp_name`: A string identifying the Grid Supply Point region.
2. `forecast_date_time`: The UTC datetime when ESO ran their forecast.
    The `forecast_date_time` will be a little time (about an hour or two?)
    after the initialisation time of the numerical weather predictions
    fed into ESO's forecasting algorithm.  This script takes the floor('30T') of the
    original forecast_date_time from ESO.  `forecast_date_time` tells us when ESO
    ran their PV forecast. The `target_date_time` (the time that each forecast is
    _about_) can be calculated as the sum of `forecast_date_time` and `step`.
    To give a little more background:  For most forecasts, any given row is identified
    by _two_ datetimes:  The datetime that the forecast was run (the `forecast_date_time`);
    and the datetime that the forecast is _about_ (the `target_date_time`).
3. `step`: The Timedelta between the forecast_date_time and the target_date_time.

It's not possible to append to NetCDF files, so this script loads everything into memory,
and strip away stuff we don't need, and maintain a list of xr.DataSets to be concatenated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# The source path of the CSV files from ESO.
ESO_CSV_PATH = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/"
    "other_organisations_pv_forecasts/National_Grid_ESO/CSV/"
)

# The destination NetCDF filename.
DST_NETCDF = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/"
    "other_organisations_pv_forecasts/National_Grid_ESO/NetCDF/ESO_GSP_PV_forecasts.nc"
)

# The DATETIME columns to load from the ESO CSVs.
DATETIME_COLS = ["FORECAST_DATE_TIME", "TARGET_DATE_TIME"]
# Ignore 'WEATHER_FORECAST_DATE_TIME'.

# The ESO forecasting algorithms (see comments a the top of this script).
ESO_ALGO_NAMES = ("ASL", "ML")


def filenames_and_datetime_periods(path: Path) -> pd.Series:
    """Gets all CSV filenames in `path`.

    Returns a Series where the Index is a pd.PeriodIndex at monthly frequency,
    and the values at the full Path to the CSV file.  The index is sorted.

    This is the header and first line of an ESO CSVs:

    ,TALENDID,WEATHER_FORECAST_DATE_TIME,FORECAST_DATE_TIME,TARGET_DATE_TIME,FORECAST_HORIZON,SITE_ID,MW,SCRIPT_NAME
    0,PV_GSP_ASL_20210401014517,2021-03-31 22:40:00+00:00,2021-04-01 00:46:36+00:00,2021-04-01 01:00:00+00:00,2D,BICF_1,0.0,GSP_PV_ASL_00.00

    TALENDID: The ID of the forecasting job running on ESO's PEF system.
    WEATHER_FORECAST_DATE_TIME: The datetime when ESO received numerical weather predictions.
        This is not exactly the same as the NWP init time.
    FORECAST_DATE_TIME: The datetime when ESO ran their PV forecasting script.
    TARGET_DATE_TIME: The time that the forecast is about.  In half hourly intervals, aligned to
        the top of the hour, and 30 minutes past the hour.
    FORECAST_HORIZON: '2D', '2d', '14D' or '14d':  ESO run a "2 day ahead" forecast and a
        "14 day ahead" forecast.
    SITE_ID: The Grid Supply Point (GSP) name.  Called the `gsp_name` in ESO's metadata.
    MW: Amount of power forecast to be produced by the SITE_ID at TARGET_DATE_TIME.
    SCRIPT_NAME: The name of ESO's forecasting script. Includes 'ASL' or 'ML'
        to indicate which ESO forecasting algorithm was used (see the comments
        at the top of this script for more info), and includes the version of the script.
    """
    csv_filenames = list(path.glob("*.csv"))

    # The filename stems are of the form "gsp_pv_forecast_Jun2020".
    # Split by "_" and then take the last split to get "Jun2020":
    periods = [filename.stem.split("_")[-1] for filename in csv_filenames]
    periods = [pd.Period(period) for period in periods]

    return pd.Series(csv_filenames, index=periods).sort_index()


def load_csv(csv_filename: Path) -> pd.DataFrame:
    """Load ESO CSV files.  Returns a pd.DataFrame."""
    eso_forecasts_df = pd.read_csv(
        csv_filename,
        usecols=DATETIME_COLS + ["FORECAST_HORIZON", "SITE_ID", "MW", "SCRIPT_NAME"],
        dtype={"MW": np.float32},
    )

    # Using `parse_dates=DATETIME_COLS` in `read_csv` takes 6 mins 15 secs on leonardo.
    # Using `to_datetime()` instead takes a total of 1 minute (read_csv then to_datetime)
    for col in DATETIME_COLS:
        eso_forecasts_df[col] = pd.to_datetime(eso_forecasts_df[col])

        # Remove timezone because xarray doesn't like timezones :)
        eso_forecasts_df[col] = eso_forecasts_df[col].dt.tz_convert("UTC").dt.tz_convert(None)

    return eso_forecasts_df


def keep_only_2_day_forecasts(eso_forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Throw away any rows where FORECAST_HORIZON != '2D'.

    ESO run two forecasts:  2D (2-day ahead) and 14D (14-day ahead).
    We only want the 2D forecasts because we're comparing against OCF's nowcasts.
    """
    eso_forecasts_df.FORECAST_HORIZON = eso_forecasts_df.FORECAST_HORIZON.str.upper()
    rows_2D_horizon = eso_forecasts_df.FORECAST_HORIZON == "2D"
    eso_forecasts_df = eso_forecasts_df[rows_2D_horizon]
    eso_forecasts_df.drop(columns="FORECAST_HORIZON", inplace=True)
    return eso_forecasts_df


def split_asl_and_ml_forecasts(eso_forecasts_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the forecasts into ASL forecasts, and ML forecasts.

    Returns a dict where the keys are the ESO forecasting algorithm name 'ASL' or 'ML',
    and the values are the pd.DataFrame of forecast data.
    """
    # The SCRIPT_NAME is the name of the ESO's PV forecasting script.
    # Includes 'ASL' or 'ML' to indicate which ESO forecasting
    # algorithm was used (see the comments at the top of this script for more info),
    # and includes the version of the script.
    # An example SCRIPT_NAME is `GSP_PV_ASL_00.00`.
    eso_forecasts_df.SCRIPT_NAME = eso_forecasts_df.SCRIPT_NAME.str.upper()

    eso_forecasts = {}
    for algo_name in ESO_ALGO_NAMES:
        mask = eso_forecasts_df.SCRIPT_NAME.str.contains(f"_{algo_name}_")
        eso_forecasts[algo_name] = eso_forecasts_df[mask]
        eso_forecasts[algo_name] = eso_forecasts[algo_name].drop(columns="SCRIPT_NAME")

    return eso_forecasts


def convert_to_dataarray(df: pd.DataFrame) -> xr.DataArray:
    """Convert forecast data to a 3-dimensional DataArray, with these dimensions:

    1. `gsp_name`: A string identifying the Grid Supply Point region.
    2. `forecast_date_time`: The UTC datetime when ESO ran their forecast.  Not exactly the
        same as the NWP init time.  This script takes the floor('30T') of the
        original forecast_date_time from ESO.
    3. `step`: The Timedelta between the forecast_date_time and the target_date_time.
    """
    df = df.copy()  # So we don't modify the passed-in `df` object.

    # The FORECAST_DATE_TIME is, I think, the time when ESO ran their forecasting script.
    # The times are all over the place.  So floor the datetime to the nearest half-hour:
    df["forecast_date_time"] = df["FORECAST_DATE_TIME"].dt.floor("30T")
    df = df.drop(columns="FORECAST_DATE_TIME")

    # Calculate the forecast "step":
    df["step"] = df["TARGET_DATE_TIME"] - df["forecast_date_time"]
    df = df.drop(columns="TARGET_DATE_TIME")

    # Make sure "step" is positive:
    # (step can be negative for ASL forecasts, for some reasons).
    df = df[df.step >= pd.Timedelta(0)]

    # Rename to more column names more like the ones we're used to.
    df = df.rename(columns={"SITE_ID": "gsp_name"})

    # Set index
    df = df.set_index(["gsp_name", "forecast_date_time", "step"])
    series = df.squeeze()
    del df

    # Remove duplicate indicies
    duplicated = series.index.duplicated()
    series = series[~duplicated]
    return series.to_xarray()


def main():
    """Loop round all the files."""
    filenames = filenames_and_datetime_periods(ESO_CSV_PATH)
    n = len(filenames)
    datasets = []

    for i, filename in enumerate(filenames.values):
        print(f"{i+1:2d}/{n:2d}: {filename}", flush=True)
        eso_forecasts_df = load_csv(filename)
        eso_forecasts_df = keep_only_2_day_forecasts(eso_forecasts_df)
        split_forecasts = split_asl_and_ml_forecasts(eso_forecasts_df)
        del eso_forecasts_df

        data_arrays = {}
        for algo_name in ESO_ALGO_NAMES:
            data_arrays[algo_name] = convert_to_dataarray(split_forecasts[algo_name])
        del split_forecasts

        dataset = xr.Dataset(data_arrays)
        datasets.append(dataset)
        del dataset, data_arrays

    dataset = xr.concat(datasets, dim="forecast_date_time")

    # The resulting NetCDF file is about 2.3 GBytes.
    dataset.to_netcdf(DST_NETCDF)


if __name__ == "__main__":
    main()
