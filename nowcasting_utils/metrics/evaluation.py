""" Evaluation the model results """

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nowcasting_utils.metrics.plot import (
    make_forecast_horizon_metrics,
    make_gsp_id_metrics,
    make_main_metrics,
    make_t0_datetime_utc_metrics,
)
from nowcasting_utils.metrics.utils import check_results_df


def evaluation(
    results_df: pd.DataFrame, model_name: str, show_fig: bool = True, save_fig: bool = True
):
    """
    Main evaluation method

    1. checks results_df is in the correct format
    2. Make evaluation and plots of the data
    3. Make evaluation and plots of the ml results
    4. Make national evaluation and plots of the ml results

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
    # make sure datetimes columns datetimes and floor target time t
    results_df["t0_datetime_utc"] = pd.to_datetime(results_df["t0_datetime_utc"])
    results_df["target_datetime_utc"] = pd.to_datetime(results_df["target_datetime_utc"])
    results_df["target_datetime_utc"] = results_df["target_datetime_utc"].dt.floor("30T")

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

    # make figure of results
    fig = results_national_evaluation(results_df, model_name)
    if show_fig:
        fig.show(renderer="browser")
    if save_fig:
        fig.write_html(f"./evaluation_results_national_{model_name}.html")


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

    trace_gsp_id, trace_gsp_id_hist = make_gsp_id_metrics(
        results_df=results_df, normalize=False, model_name=model_name
    )
    trace_gsp_id_normalized, trace_gsp_id_normalized_hist = make_gsp_id_metrics(
        results_df=results_df, normalize=True, model_name=model_name
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


def results_national_evaluation(results_df: pd.DataFrame, model_name: str) -> go.Figure:
    """
    Calculate metrics of the results

    Args:
        results_df: results dataframe
        model_name: the model name, used for adding titles to plots

    """
    # lets just get unique results on three columns
    results_df = (
        results_df.groupby(["t0_datetime_utc", "target_datetime_utc", "gsp_id"])
        .first()
        .reset_index()
    )

    national_results_df = (
        results_df.groupby(["t0_datetime_utc", "target_datetime_utc"]).sum().reset_index()
    )
    national_results_df_count = (
        results_df.groupby(["t0_datetime_utc", "target_datetime_utc"]).count().reset_index()
    )
    national_results_df["gsp_id_count"] = national_results_df_count["gsp_id"]

    # *******
    # main metrics (and normalized)
    # ********

    trace_main_national = make_main_metrics(results_df=national_results_df, normalize=False)
    trace_main_national_normalized = make_main_metrics(
        results_df=national_results_df, normalize=True
    )

    # *******
    # evaluate per forecast horizon
    # ********

    trace_forecast_horizons_national = make_forecast_horizon_metrics(
        results_df=national_results_df, normalize=False
    )
    trace_forecast_horizons_national_normalized = make_forecast_horizon_metrics(
        results_df=national_results_df, normalize=True
    )

    # *******
    # evaluate by "t0_datetime_utc"
    # *******

    trace_datetime, trace_datetime_hist = make_t0_datetime_utc_metrics(
        results_df=national_results_df, normalize=False
    )
    trace_datetime_normalized, trace_datetime_normalized_hist = make_t0_datetime_utc_metrics(
        results_df=national_results_df, normalize=True
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
            "National Metrics",
            "Normalised National Metrics",
            "National Forecast Error as a function of Forecast Horizon",
            "National Forecast Normalised Error as a function of Forecast Horizon",
            "National Forecast Error as a function of time",
            "National Forecast Normalised Error as a function of time",
            "Histogram of National MAE for over time",
            "Histogram of National NMAE for over time",
        ],
    )

    fig.append_trace(trace_main_national, 1, 1)
    fig.append_trace(trace_main_national_normalized, 1, 2)
    for trace in trace_forecast_horizons_national:
        fig.append_trace(trace, 2, 1)
    for trace in trace_forecast_horizons_national_normalized:
        fig.append_trace(trace, 2, 2)
    for trace in trace_datetime:
        fig.append_trace(trace, 3, 1)
    for trace in trace_datetime_normalized:
        fig.append_trace(trace, 3, 2)
    fig.append_trace(trace_datetime_hist, 4, 1)
    fig.append_trace(trace_datetime_normalized_hist, 4, 2)

    fig["layout"]["xaxis1"]["title"] = "Forecast horizon (hours)"
    fig["layout"]["xaxis2"]["title"] = "Forecast horizon (hours)"
    fig["layout"]["xaxis3"]["title"] = "time"
    fig["layout"]["xaxis4"]["title"] = "time"
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

    # find out if there are any empty gsp with no data
    gsp_ids = sorted(results_df["gsp_id"].unique())
    empty_gsp = [x for x in range(1, 339) if x not in gsp_ids]
    print(f"GSP with no data are {empty_gsp}")

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
