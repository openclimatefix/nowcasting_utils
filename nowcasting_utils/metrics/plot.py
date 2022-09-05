""" Plotting functions """
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
from plotly import graph_objects as go

colours = ["rgb(77,137,99)", "rgb(225,179,120)", "rgb(224,204,151)", "rgb(105,165,131)"]
plotting_metrics = ["MAE"]


def make_main_metrics(results_df, normalize: bool = False) -> go.Table:
    """Make whole metrics and make plotly Table"""
    y_hat = results_df["forecast_gsp_pv_outturn_mw"]
    y = results_df["actual_gsp_pv_outturn_mw"]

    if normalize:
        y_hat = y_hat / results_df["capacity_mwp"]
        y = y / results_df["capacity_mwp"]

    main_metrics = run_metrics(y=y, y_hat=y_hat, name="All horizons")

    if "gsp_id_count" in results_df.keys():
        main_metrics["Average GSP for each forecast"] = results_df["gsp_id_count"].mean()

    # metrics ready for plotting
    metrics = list(main_metrics.keys())
    if normalize:
        metrics = [f"Normalised {metric}" for metric in metrics]
        main_metrics["Number of Data Points"] = main_metrics["Number of Data Points"] / 100
        if "gsp_id_count" in results_df.keys():
            main_metrics["Average GSP for each forecast"] = (
                main_metrics["Average GSP for each forecast"] / 100
            )

    # values ready for plotting
    values = np.array(list(main_metrics.values()))
    if normalize:
        values *= 100
    values = np.round(values, 3)

    if normalize:
        header = ["Normalised Metric", "Values [%]"]
    else:
        header = ["Metric", "Values [MW]"]

    # make table for plot
    trace_main_normalized = go.Table(
        header=dict(values=header),
        cells=dict(values=[metrics, values]),
    )

    return trace_main_normalized


def make_forecast_horizon_metrics(results_df, normalize: bool = False) -> List[go.Trace]:
    """
    Make forecast horizons metrics and plots
    """
    n_forecast_hoirzons = 4
    time_delta = timedelta(minutes=30)
    forecast_horizon_metrics = {}
    # loop over the number of forecast horizons
    for i in range(n_forecast_hoirzons):

        forecast_horizon = (i + 1) * time_delta
        forecast_horizon_hours = forecast_horizon.seconds / 3600

        results_df_one_forecast_hoirzon = results_df[
            results_df["target_datetime_utc"] - results_df["t0_datetime_utc"]
            == (i + 1) * time_delta
        ]

        y_hat = results_df_one_forecast_hoirzon["forecast_gsp_pv_outturn_mw"]
        y = results_df_one_forecast_hoirzon["actual_gsp_pv_outturn_mw"]

        if normalize:
            y_hat = 100 * y_hat / results_df["capacity_mwp"]
            y = 100 * y / results_df["capacity_mwp"]

        if normalize:
            legendgroup = "2"
        else:
            legendgroup = "1"

        forecast_horizon_metrics[forecast_horizon_hours] = run_metrics(
            y=y, y_hat=y_hat, name=f"Forecast horizon: {forecast_horizon}"
        )

    forecast_horizon_metrics_df = pd.DataFrame(forecast_horizon_metrics).T
    trace_forecast_horizons = []
    # loop over the different columns / metrics and plot them
    for i in range(len(plotting_metrics)):

        col = plotting_metrics[i]
        colour = colours[i]

        name = col
        if normalize:
            name = f"Normalised {col}"

        trace_forecast_horizon = go.Scatter(
            x=forecast_horizon_metrics_df.index,
            y=forecast_horizon_metrics_df[col],
            name=name,
            legendgroup=legendgroup,
            marker=dict(color=colour),
        )
        trace_forecast_horizons.append(trace_forecast_horizon)

    return trace_forecast_horizons


def make_gsp_id_metrics(
    results_df, model_name: str, normalize: bool = False
) -> (go.Scatter, go.Histogram):
    """
    Make the gsp id metrics

    1. make metrics
    2. plot metrics per gsp
    3. make histogram of MAE
    """

    # make metrics per gsp
    n_gsp_ids = int(results_df["gsp_id"].max())
    gsp_metrics = {}
    for i in range(n_gsp_ids):

        gsp_id = i + 1

        results_df_one_forecast_hoirzon = results_df[results_df["gsp_id"] == gsp_id]

        y_hat = results_df_one_forecast_hoirzon["forecast_gsp_pv_outturn_mw"]
        y = results_df_one_forecast_hoirzon["actual_gsp_pv_outturn_mw"]

        if normalize:
            y_hat = 100 * y_hat / results_df["capacity_mwp"]
            y = 100 * y / results_df["capacity_mwp"]

        gsp_metrics[gsp_id] = run_metrics(y=y, y_hat=y_hat, name=f"GSP ID: {gsp_id}")

    gsp_metrics_df = pd.DataFrame(gsp_metrics).T

    # plot metrics
    trace_gsp_id = []
    for i in range(len(plotting_metrics)):

        col = plotting_metrics[i]
        colour = colours[i]

        name = col
        if normalize:
            name = f"Normalised {col}"

        trace_forecast_horizon = go.Scatter(
            x=gsp_metrics_df.index,
            y=gsp_metrics_df[col],
            name=name,
            showlegend=False,
            marker=dict(color=colour),
            line={"shape": "hv"},
        )
        trace_gsp_id.append(trace_forecast_horizon)

    # make histogram
    trace_histogram = go.Histogram(
        x=gsp_metrics_df["MAE"],
        marker_color=colours[0],
        showlegend=False,
    )

    # save to csv
    if normalize:
        model_name = f"{model_name}_normalized"
    gsp_metrics_df.to_csv(f"{model_name}.csv")

    return trace_gsp_id, trace_histogram


def make_t0_datetime_utc_metrics(results_df, normalize: bool = False) -> (go.Scatter, go.Histogram):
    """
    Make the datetime metrics

    1. make metrics
    2. plot metrics per datetime
    3. make histogram of MAE
    """

    # make metrics per gsp
    target_datetimes_utc = sorted(results_df["target_datetime_utc"].unique())
    t0_datetime_metrics = {}

    results_df = results_df.copy()
    if normalize:
        results_df["forecast_gsp_pv_outturn_mw"] = (
            100 * results_df["forecast_gsp_pv_outturn_mw"] / results_df["capacity_mwp"]
        )
        results_df["actual_gsp_pv_outturn_mw"] = (
            100 * results_df["actual_gsp_pv_outturn_mw"] / results_df["capacity_mwp"]
        )

    for i in range(len(target_datetimes_utc)):

        target_datetime_utc = target_datetimes_utc[i]

        results_df_one_datetime = results_df[
            results_df["target_datetime_utc"] == target_datetime_utc
        ]

        y_hat = results_df_one_datetime["forecast_gsp_pv_outturn_mw"]
        y = results_df_one_datetime["actual_gsp_pv_outturn_mw"]

        t0_datetime_metrics[target_datetime_utc] = run_metrics(
            y=y, y_hat=y_hat, name=f"target_datetime_utc: {target_datetime_utc}"
        )

    gsp_metrics_df = pd.DataFrame(t0_datetime_metrics).T

    # plot metrics
    trace_gsp_id = []
    for i in range(len(plotting_metrics)):

        col = plotting_metrics[i]
        colour = colours[i]

        name = col
        if normalize:
            name = f"Normalised {col}"

        trace_forecast_horizon = go.Scatter(
            x=gsp_metrics_df.index,
            y=gsp_metrics_df[col],
            name=name,
            showlegend=False,
            marker=dict(color=colour),
            line={"shape": "hv"},
        )
        trace_gsp_id.append(trace_forecast_horizon)

    # make histogram
    trace_histogram = go.Histogram(
        x=gsp_metrics_df["MAE"],
        marker_color=colours[0],
        showlegend=False,
    )

    return trace_gsp_id, trace_histogram


def run_metrics(y_hat: pd.Series, y: pd.Series, name: str) -> dict:
    """
    Make metrics from truth and predictions
    """

    # basic metrics
    mean_absolute_error = (y - y_hat).abs().mean()
    mean_error = (y - y_hat).mean()
    root_mean_squared_error = (((y - y_hat) ** 2).mean()) ** 0.5

    # max value
    max_absolute_error = (y - y_hat).max()

    # std and CI statistics
    std_absolute_error = (y - y_hat).abs().std()
    n_data = len(y)
    if n_data > 0:
        ci_absolute_error = std_absolute_error / (n_data**0.5)
    else:
        ci_absolute_error = np.nan

    metrics = {
        "MAE": mean_absolute_error,
        "Mean Error": mean_error,
        "RMSE": root_mean_squared_error,
        "Max Absolute Error": max_absolute_error,
        "Std Absolute Error": std_absolute_error,
        "Ci Absolute Error": ci_absolute_error,
        "Number of Data Points": n_data,
    }

    return metrics
