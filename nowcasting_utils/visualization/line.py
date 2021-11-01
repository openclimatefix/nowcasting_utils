"""Several line plots of predictions and truths."""

from typing import List, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_trace(x, y, truth: bool, show_legend: bool = True):
    """
    Make a plotly trace data (x,y).

    Args:
        x: time series of data
        y: values of data
        truth: if y is the truth or predictions. The colour of the line changed depending on this
        show_legend: option to show the legend for this trace or not.

    Returns:plotly trace

    """
    color = "Blue" if truth else "Red"
    name = "truth" if truth else "predict"

    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        marker=dict(color=color, line=dict(color=color, width=2)),
        name=name,
        showlegend=show_legend,
    )


def plot_one_result(x, y, y_hat):
    """
    Plot one result.

    Args:
        x: time series for the forecast predictions, should be size [forecast length]
        y: the truth values of pv yield, should be size [forecast length]
        y_hat: the predicted values of pv yield, should be size [forecast length]

    Returns:a plotly figure

    """
    fig = go.Figure()
    fig.add_trace(make_trace(x=x, y=y, truth=True))
    fig.add_trace(make_trace(x=x, y=y_hat, truth=False))

    return fig


# plot all of batch
def plot_batch_results(
    x: Union[np.array, List],
    y: np.array,
    y_hat: np.array,
    model_name: str,
    x_hat: Union[np.array, List] = None,
):
    """
    Plot batch results.

    Args:
        x: is a list of time series for the different predictions in the batch,
            should be size [batch_size, forecast length]
        y: the truth values of pv yield, should be size [batch_size, forecast length]
        y_hat: the predicted values of pv yield, should be size [batch_size, forecast length]
        model_name: the name of the model
        x_hat: the x values for the predictions (y_hat),
            note that if none is supplied then x is used instead

    Returns: a plotly figure
    """
    if x_hat is None:
        x_hat = x

    batchsize = y.shape[0]
    N = int(np.ceil(batchsize ** 0.5))

    subplot_titles = [str(i) for i in range(batchsize)]

    fig = make_subplots(
        rows=N,
        cols=N,
        subplot_titles=subplot_titles,
        x_title=f"Batch Plot of PV Predict: {model_name}",
    )

    # move the x_title to the top
    # Could perhaps do this in a neater way,
    # just happens that the last annotation is the x_title object
    fig.layout.annotations[-1]["y"] = 1
    fig.layout.annotations[-1]["yshift"] = 30
    fig.layout.annotations[-1]["yanchor"] = "bottom"

    for i in range(0, batchsize):

        row = i // N + 1
        col = i % N + 1

        fig.add_trace(
            make_trace(x=x[i], y=y[i], truth=True, show_legend=False if i > 0 else True),
            row=row,
            col=col,
        )
        fig.add_trace(
            make_trace(x=x_hat[i], y=y_hat[i], truth=False, show_legend=False if i > 0 else True),
            row=row,
            col=col,
        )
    fig.update_layout(
        width=1500,
        height=1500,
    )
    fig.update_yaxes(range=[0, 1])

    return fig
