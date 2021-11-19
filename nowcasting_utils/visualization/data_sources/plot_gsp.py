""" General functions for plotting PV data """
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from plotly.subplots import make_subplots

from nowcasting_utils.visualization.line import make_trace
from nowcasting_utils.visualization.utils import make_buttons, make_slider


def get_trace_centroid_gsp(gsp: GSP, example_index: int) -> go.Scatter:
    """Produce plot of centroid GSP"""

    y = gsp.power_normalized[example_index, :, 0]
    x = gsp.time[example_index]

    return make_trace(x, y, truth=True, name="center gsp", color="Blue")


def get_trace_all_gsps(gsp: GSP, example_index: int) -> List[go.Scatter]:
    """Produce plot of centroid GSP"""

    traces = []
    x = gsp.time[example_index]
    n_gsps = gsp.power_mw.shape[2]

    # make the lines a little bit see-through
    opacity = (1 / n_gsps) ** 0.25

    for gsp_index in range(1, n_gsps):
        y = gsp.power_normalized[example_index, :, gsp_index]

        gsp_id = gsp.id[example_index, gsp_index].values
        truth = False

        if ~np.isnan(gsp_id):
            gsp_id = int(gsp_id)
            name = f"GSP {gsp_id}"

            traces.append(make_trace(x, y, truth=truth, name=name, color="Green", opacity=opacity))

    centroid_trace = get_trace_centroid_gsp(gsp=gsp, example_index=example_index)
    centroid_trace["legendrank"] = 1
    traces.append(centroid_trace)

    return traces


def get_traces_gsp_intensity(gsp: GSP, example_index: int):
    """Get traces of pv intenisty map"""
    time = gsp.time[example_index]

    traces = []

    for t_index in range(len(time)):

        trace = get_trace_gsp_intensity_one_time_step(
            gsp=gsp, example_index=example_index, t_index=t_index
        )
        traces.append(trace)

    return traces


def get_trace_gsp_intensity_one_time_step(gsp: GSP, example_index: int, t_index: int):
    """Get trace of pv intensity map"""
    time = gsp.time[example_index]
    x = gsp.x_coords[example_index]
    y = gsp.y_coords[example_index]
    gsp_id = gsp.id[example_index].values

    n_gsp_systems = gsp.power_mw.shape[2]

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    z = gsp.power_normalized[example_index, t_index, :]
    name = time[t_index].data
    z = z.fillna(0)

    if z.max() > 0:
        size = 200 * z
    else:
        size = 0

    lat = np.round(lat, 4)
    lon = np.round(lon, 4)

    text = [f"GSP {gsp_id}: {z.values:.2f}" for z, gsp_id in zip(z, gsp_id)]

    # TODO change this to use GSP boundaries #55
    trace = go.Scattermapbox(
        lat=lat,
        lon=lon,
        marker=dict(color=["Blue"] + ["Green"] * (n_gsp_systems - 1), size=size, sizemode="area"),
        name=str(name),
        text=text,
    )

    return trace


def make_fig_of_animation_from_frames(traces, gsp: GSP, example_index: int):
    """Make animated fig form traces"""

    frames = []
    for i, trace in enumerate(traces[1:]):
        frames.append(go.Frame(data=trace, name=f"frame{i+1}"))

    # make slider
    labels = [pd.to_datetime(time.data) for time in gsp.time[example_index]]
    sliders = make_slider(labels=labels)

    x = gsp.x_coords[example_index][gsp.x_coords[example_index] != 0].mean()
    y = gsp.y_coords[example_index][gsp.y_coords[example_index] != 0].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    fig = go.Figure(
        data=traces[0],
        layout=go.Layout(
            title="Start Title",
        ),
        frames=frames,
    )
    fig.update_layout(updatemenus=[make_buttons()])
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=8, mapbox_center={"lat": lat, "lon": lon}
    )

    fig.update_layout(sliders=sliders)

    return fig


def get_fig_gsp_combined(gsp: GSP, example_index: int):
    """
    Create a combined plot

    1. Plot the gsp intensity in time
    2. Plot the gsp intensity with coords and animate in time
    """

    traces_pv_intensity_in_time = get_trace_all_gsps(gsp=gsp, example_index=example_index)

    traces_pv_intensity_map = get_traces_gsp_intensity(gsp=gsp, example_index=example_index)

    x = gsp.x_coords[example_index][gsp.x_coords[example_index] != 0].mean()
    y = gsp.y_coords[example_index][gsp.y_coords[example_index] != 0].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Map", "Time Series"),
        specs=[
            [{"type": "choroplethmapbox"}, {"type": "xy"}],
        ],
    )

    # add first animation plot
    fig.add_trace(trace=traces_pv_intensity_map[0], row=1, col=1)

    # add all time series plots
    for trace in traces_pv_intensity_in_time:
        fig.add_trace(trace, row=1, col=2)

    n_traces = len(fig.data)

    frames = []
    static_traces = list(fig.data[1:])
    for i, trace in enumerate(traces_pv_intensity_map):
        frames.append(
            dict(data=[trace] + static_traces, traces=list(range(n_traces)), name=f"frame{i}")
        )

    # make slider
    labels = [pd.to_datetime(time.data) for time in gsp.time[example_index]]
    sliders = make_slider(labels=labels)

    fig.update(frames=frames)
    fig.update_layout(updatemenus=[make_buttons()])

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=8, mapbox_center={"lat": lat, "lon": lon}
    )

    fig.update_layout(sliders=sliders)

    return fig
