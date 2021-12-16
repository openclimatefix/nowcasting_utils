""" General functions for plotting PV data """
from typing import List

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nowcasting_dataset.data_sources.gsp.gsp_model import GSP
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from plotly.subplots import make_subplots

from nowcasting_utils.visualization.line import make_trace
from nowcasting_utils.visualization.utils import make_buttons, make_slider

from nowcasting_dataset.data_sources.gsp.eso import (
    get_gsp_shape_from_eso,
    get_gsp_metadata_from_eso,
)

WGS84_CRS = "EPSG:4326"


def get_trace_centroid_gsp(gsp: GSP, example_index: int) -> go.Scatter:
    """Produce plot of centroid GSP"""

    y = gsp.power_normalized[example_index, :, 0]
    x = gsp.time[example_index]

    return make_trace(x, y, truth=True, name="GSP", color="Blue")


def get_trace_all_gsps(
    gsp: GSP, example_index: int, plot_other_gsp: bool = False
) -> List[go.Scatter]:
    """Produce plot of centroid GSP"""

    traces = []
    x = gsp.time[example_index]
    if plot_other_gsp:
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

                traces.append(
                    make_trace(
                        x, y, truth=truth, name=name, color="Green", opacity=opacity, mode="lines"
                    )
                )

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
    gsp_id = gsp.id[example_index].values
    name = str(time[t_index].data)

    # get shape from eso
    gsp_metadata = get_gsp_metadata_from_eso()
    gsp_metadata = gsp_metadata.to_crs(WGS84_CRS)

    # select first GSP system
    gsp_data_to_plot = gsp_metadata
    gsp_data_to_plot = gsp_data_to_plot[gsp_data_to_plot["gsp_id"] == gsp_id[0]]

    gsp_data_to_plot["Amount"] = gsp.power_normalized[example_index, t_index, 0].values

    shapes_dict = json.loads(gsp_data_to_plot.to_json())

    trace = go.Choroplethmapbox(
        geojson=shapes_dict,
        locations=gsp_data_to_plot.index,
        z=gsp_data_to_plot.Amount,
        colorscale="Viridis",
        name=name,
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
