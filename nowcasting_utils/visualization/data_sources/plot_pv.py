""" General functions for plotting PV data """
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from nowcasting_dataset.data_sources.pv.pv_data_source import PV
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from plotly.subplots import make_subplots

from nowcasting_utils.visualization.line import make_trace
from nowcasting_utils.visualization.utils import make_buttons, make_slider


def get_trace_centroid_pv(pv: PV, example_index: int) -> go.Scatter:
    """Produce plot of centroid pv system"""

    y = pv.power_normalized[example_index, :, 0]
    x = pv.time[example_index]

    return make_trace(x, y, truth=True, name="centorid pv",mode='lines')


def get_trace_all_pv_systems(
    pv: PV, example_index: int, center_system: bool = True
) -> List[go.Scatter]:
    """Produce plot of centroid pv system"""

    traces = []
    x = pv.time[example_index]
    n_pv_systems = pv.power_mw.shape[2]
    print(pv.power_mw.shape)
    print(n_pv_systems)

    if center_system:
        start_idx = 1
        centroid_trace = get_trace_centroid_pv(pv=pv, example_index=example_index)
        traces.append(centroid_trace)

    else:
        start_idx = 0

    # make the lines a little bit see-through
    opacity = (1 / n_pv_systems) ** 0.35

    for pv_system_index in range(start_idx, n_pv_systems):
        y = pv.power_normalized[example_index, :, pv_system_index]

        pv_id = pv.id[example_index, pv_system_index].values
        truth = False

        if ~np.isnan(pv_id):
            pv_id = int(pv_id)
            name = f"PV system {pv_id}"
            traces.append(make_trace(x, y, truth=truth, name=name, opacity=opacity, mode='lines'))

    return traces


def get_traces_pv_intensity(pv: PV, example_index: int):
    """Get traces of pv intensity map"""
    time = pv.time[example_index]

    traces = [go.Choroplethmapbox(colorscale="Viridis")]

    for t_index in range(len(time)):

        trace = get_trace_pv_intensity_one_time_step(
            pv=pv, example_index=example_index, t_index=t_index
        )

        traces.append(trace)

    return traces


def get_trace_pv_intensity_one_time_step(
    pv: PV, example_index: int, t_index: int, center: bool = False
):
    """Get trace of pv intensity map"""
    time = pv.time[example_index]
    x = pv.x_coords[example_index]
    y = pv.y_coords[example_index]
    pv_id = pv.id[example_index].values

    n_pv_systems = pv.power_mw.shape[2]

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    z = pv.power_normalized[example_index, t_index, :]
    name = time[t_index].data

    if center:
        colour = (["Blue"] + ["Red"] * (n_pv_systems - 1),)
    else:
        colour = ["Red"] * n_pv_systems

    z = z.fillna(0)
    size = 200 * z

    lat = np.round(lat, 4)
    lon = np.round(lon, 4)

    text = [f"PV {pv_id}: {z.values:.2f}" for z, pv_id in zip(z, pv_id)]

    trace = go.Scattermapbox(
        lat=lat,
        lon=lon,
        marker=dict(color=colour, size=size, sizemode="area"),
        name=str(name),
        text=text,
    )

    return trace


def make_fig_of_animation_from_frames(traces, pv, example_index):
    """Make animated fig form traces"""

    frames = []
    for i, trace in enumerate(traces[1:]):
        frames.append(go.Frame(data=trace, name=f"frame{i+1}"))

    # make slider
    labels = [pd.to_datetime(time.data) for time in pv.time[example_index]]
    sliders = make_slider(labels=labels)

    x = pv.x_coords[example_index][pv.x_coords[example_index] != 0].mean()
    y = pv.y_coords[example_index][pv.y_coords[example_index] != 0].mean()

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


def get_fig_pv_combined(pv: PV, example_index: int):
    """
    Create a combined plot

    1. Plot the pv intensity in time
    2. Plot the pv intensity with coords and animate in time
    """

    traces_pv_intensity_in_time = get_trace_all_pv_systems(
        pv=pv, example_index=example_index, center_system=False
    )

    traces_pv_intensity_map = get_traces_pv_intensity(pv=pv, example_index=example_index)

    x = pv.x_coords[example_index][pv.x_coords[example_index] != 0].mean()
    y = pv.y_coords[example_index][pv.y_coords[example_index] != 0].mean()

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
    labels = [pd.to_datetime(time.data) for time in pv.time[example_index]]
    sliders = make_slider(labels=labels)

    fig.update(frames=frames)
    fig.update_layout(updatemenus=[make_buttons()])

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=8, mapbox_center={"lat": lat, "lon": lon}
    )

    fig.update_layout(sliders=sliders)

    return fig
