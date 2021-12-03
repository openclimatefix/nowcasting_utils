""" Evaluation the model results """
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nowcasting_utils.metrics.utils import check_results_df

colours = ["rgb(77,137,99)", "rgb(225,179,120)", "rgb(224,204,151)", "rgb(105,165,131)"]


def evaluation(
    results_df: pd.DataFrame, model_name: str, show_fig: bool = True, save_fig: bool = True
):
    """
    Main evaluation method

    1. checks results_df is in the correct format
    2. Make evaluation and plots of the data
    3. Make evaluation and plots of the ml results

    Args:
        results_df: results dataframe. This should have the following columns:
            - t0_datetime_utc
            - target_datetime_utc
            - forecast_gsp_pv_outturn_mw
            - actual_gsp_pv_outturn_mw
            - gsp_id
            - capacity_mwp
        model_name: the model name, used for adding titles to plots
        show_fig: display figure in browser - this doesnt work for CI
        save_fig: option to save figure or not

    """
    # make sure datetimes columns datetimes
    results_df["t0_datetime_utc"] = pd.to_datetime(results_df["t0_datetime_utc"])
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["target_datetime_utc"])

    # check result format
    check_results_df(results_df)

    # make figure of data
    fig = data_evaluation(results_df, model_name)
    if show_fig:
        fig.show(renderer="browser")
    if save_fig:
        fig.write_html(f"./evaluation_data_{model_name}.html")

    # make figure of results
    fig = results_evaluation(results_df, model_name)
    if show_fig:
        fig.show(renderer="browser")
    if save_fig:
        fig.write_html(f"./evaluation_results_{model_name}.html")


def results_evaluation(results_df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    Calculate metrics of the results

    Args:
        results_df: results dataframe
        model_name: the model name, used for adding titles to plots

    """

    # *******
    # main metrics (and normalized)
    # ********

    trace_main = make_main_metrics(results_df=results_df, normalize=False)
    trace_main_normalized = make_main_metrics(results_df=results_df, normalize=True)

    # *******
    # evaluate per forecast horizon
    # ********

    trace_forecast_horizons = make_forecast_horizon_metrics(results_df=results_df, normalize=False)
    trace_forecast_horizons_normalized = make_forecast_horizon_metrics(
        results_df=results_df, normalize=True
    )

    # *******
    # evaluate per gsp
    # ********

    trace_gsp_id, trace_gsp_id_hist = make_gsp_id_metrics(results_df=results_df, normalize=False)
    trace_gsp_id_normalized, trace_gsp_id_normalized_hist = make_gsp_id_metrics(
        results_df=results_df, normalize=True
    )

    # *******
    # plot
    # ********
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            [{"type": "table"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=[
            "Metrics",
            "Normalised Metrics",
            "Forecast Error as a function of Forecast Horizon",
            "Forecast Normalised Error as a function of Forecast Horizon",
            "Forecast Error as a function of GSP ID",
            "Forecast Normalised Error as a function of GSP ID",
            "Histogram of MAE for each GSP",
            "Histogram of NMAE for each GSP",
        ],
    )

    fig.append_trace(trace_main, 1, 1)
    fig.append_trace(trace_main_normalized, 1, 2)
    for trace in trace_forecast_horizons:
        fig.append_trace(trace, 2, 1)
    for trace in trace_forecast_horizons_normalized:
        fig.append_trace(trace, 2, 2)
    for trace in trace_gsp_id:
        fig.append_trace(trace, 3, 1)
    for trace in trace_gsp_id_normalized:
        fig.append_trace(trace, 3, 2)
    fig.append_trace(trace_gsp_id_hist, 4, 1)
    fig.append_trace(trace_gsp_id_normalized_hist, 4, 2)

    fig["layout"]["xaxis1"]["title"] = "Forecast horizon (hours)"
    fig["layout"]["xaxis2"]["title"] = "Forecast horizon (hours)"
    fig["layout"]["xaxis3"]["title"] = "GSP ID"
    fig["layout"]["xaxis4"]["title"] = "GSP ID"
    fig["layout"]["xaxis5"]["title"] = "MAE [MW]"
    fig["layout"]["xaxis6"]["title"] = "NMAE [%]"
    fig["layout"]["yaxis1"]["title"] = "Metric [MW]"
    fig["layout"]["yaxis2"]["title"] = "Metric [%]"
    fig["layout"]["yaxis3"]["title"] = "Metric [MW]"
    fig["layout"]["yaxis4"]["title"] = "Metric [%]"
    fig["layout"]["yaxis5"]["title"] = "Count"
    fig["layout"]["yaxis6"]["title"] = "Count"
    fig["layout"]["title"] = model_name
    fig.update_layout(xaxis_range=[0.4, 2.1])
    fig.update_layout(xaxis2_range=[0.4, 2.1])

    return fig


def make_main_metrics(results_df, normalize: bool = False) -> go.Table:
    """Make whole metrics and make plotly Table"""
    y_hat = results_df["forecast_gsp_pv_outturn_mw"]
    y = results_df["actual_gsp_pv_outturn_mw"]

    if normalize:
        y_hat = y_hat / results_df["capacity_mwp"]
        y = y / results_df["capacity_mwp"]

    main_metrics = run_metrics(y=y, y_hat=y_hat, name="All horizons")

    # metrics ready for plotting
    metrics = list(main_metrics.keys())
    if normalize:
        metrics = [f"Normalised {metric}" for metric in metrics]

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
    for i in range(len(forecast_horizon_metrics_df.columns)):

        col = forecast_horizon_metrics_df.columns[i]
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


def make_gsp_id_metrics(results_df, normalize: bool = False) -> (go.Scatter, go.Histogram):
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
    for i in range(len(gsp_metrics_df.columns)):

        col = gsp_metrics_df.columns[i]
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
        x=gsp_metrics_df["Mean Absolute Error"],
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
    if len(y) > 0:
        ci_absolute_error = std_absolute_error / (len(y) ** 0.5)
    else:
        ci_absolute_error = np.nan

    # print metrics out
    print(name)
    print(f"{mean_absolute_error=:0.3f} MW (+- {ci_absolute_error:0.3f})")
    print(f"{mean_error=:0.3f} MW")
    print(f"{root_mean_squared_error=:0.3f} MW")

    print(f"{max_absolute_error=:0.3f} MW")

    print("")

    return {
        "Mean Absolute Error": mean_absolute_error,
        # "Mean Error": mean_error,
        "Root Mean Squared Error": root_mean_squared_error,
        # "Max Absolute Error": max_absolute_error,
        # "Std Absolute Error": std_absolute_error,
        # "Ci Absolute Error": ci_absolute_error,
    }


def data_evaluation(results_df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    Calculate metrics of the data in the results

    Args:
        results_df: results dataframe
        model_name: the name of the model, used as a plot title

    """

    N_data = len(results_df)

    print(f"Number of data points: {N_data}")

    # set up plots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Histogram of data distribution as a function of time",
            "Histogram of data distribution as a function of GSP ID",
        ],
    )

    # plot histogram of datetimes
    months = results_df["t0_datetime_utc"].dt.strftime("%Y-%m")
    trace0 = go.Histogram(x=months, showlegend=False)

    # plot histogram of gsp_ids
    N_bins = int(results_df["gsp_id"].max() - results_df["gsp_id"].min() + 1)
    trace1 = go.Histogram(x=results_df["gsp_id"], nbinsx=N_bins, showlegend=False)

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)

    fig["layout"]["xaxis"]["title"] = "Time"
    fig["layout"]["xaxis2"]["title"] = "GSP ID"
    fig["layout"]["yaxis"]["title"] = "Count"
    fig["layout"]["yaxis2"]["title"] = "Count"
    fig["layout"]["title"] = model_name

    return fig
