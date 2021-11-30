""" Evaluation the model results """
import pandas as pd
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nowcasting_utils.metrics.utils import check_results_df


def evaluation(results_df: pd.DataFrame):
    """
    Main evaluation method

    args:
        results_df: results dataframe

    """

    check_results_df(results_df)

    data_evaluation(results_df)

    results_evaluation(results_df)


def results_evaluation(results_df: pd.DataFrame):
    """
    Calculate metrics of the results

    args:
        results_df: results dataframe

    """

    # *******
    # main metrics
    # ********

    y_hat = results_df["forecast_gsp_pv_outturn_mw"]
    y = results_df["actual_gsp_pv_outturn_mw"]

    run_metrics(y=y, y_hat=y_hat, name='All horizons')

    # *******
    # evaluate per forecast horizon
    # ********

    n_forecast_hoirzons = 4
    time_delta = timedelta(minutes=30)
    for i in range(n_forecast_hoirzons):

        forecast_horizon = (i + 1) * time_delta

        results_df_one_forecast_hoirzon = results_df[
            results_df["target_datetime_utc"] - results_df["t0_datetime_utc"]
            == (i + 1) * time_delta
        ]

        y_hat = results_df_one_forecast_hoirzon["forecast_gsp_pv_outturn_mw"]
        y = results_df_one_forecast_hoirzon["actual_gsp_pv_outturn_mw"]

        run_metrics(y=y, y_hat=y_hat, name=f'Forecast horizon: {forecast_horizon}')


def run_metrics(y_hat: pd.Series, y: pd.Series, name:str):

    mean_absolute_error = (y - y_hat).abs().mean()
    mean_error = (y - y_hat).mean()
    root_mean_squared_error = (((y - y_hat) ** 2).mean()) ** 0.5

    max_absolute_error = (y - y_hat).max()

    std_absolute_error = (y - y_hat).abs().std()
    ci_absolute_error = std_absolute_error/(len(y)**0.5)

    print(name)
    print(f"{mean_absolute_error=:0.3f} MW (+- {ci_absolute_error:0.3f})")
    print(f"{mean_error=:0.3f} MW")
    print(f"{root_mean_squared_error=:0.3f} MW")

    print(f"{max_absolute_error=:0.3f} MW")

    print('')

    return {'mean_absolute_error': mean_absolute_error,
            'mean_error': mean_error,
            'root_mean_squared_error': root_mean_squared_error,
            'max_absolute_error': max_absolute_error,
            'std_absolute_error': std_absolute_error,
            'ci_absolute_error': ci_absolute_error
            }


def data_evaluation(results_df: pd.DataFrame):
    """
    Calculate metrics of the data in the results

    args:
        results_df: results dataframe

    """

    N_data = len(results_df)

    print(f"Number of data points: {N_data}")

    # set up plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Time distribution','GSP ID distribution'])

    # plot histogram of datetimes
    months = results_df['t0_datetime_utc'].dt.strftime('%Y-%m')
    trace0 = go.Histogram(x=months)

    # plot histogram of gsp_ids
    N_bins = int(results_df['gsp_id'].max() - results_df['gsp_id'].min() + 1)
    trace1 = go.Histogram(x=results_df['gsp_id'],nbinsx=N_bins)

    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)

    fig['layout']['xaxis']['title'] = 'Time'
    fig['layout']['xaxis2']['title'] = 'GSP ID'
    fig['layout']['yaxis']['title'] = 'Count'
    fig['layout']['yaxis2']['title'] = 'Count'

    fig.show(renderer='browser')

