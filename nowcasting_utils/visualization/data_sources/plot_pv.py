""" General functions for plotting PV data """
from typing import List

import pandas as pd
import plotly.graph_objects as go
from nowcasting_dataset.data_sources.pv.pv_data_source import PV
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from plotly.subplots import make_subplots

from nowcasting_utils.visualization.line import make_trace


def get_trace_centroid_pv(pv: PV, example_index: int) -> go.Scatter:
    """Produce plot of centroid pv system"""

    y = pv.data[example_index, :, 0]
    x = pv.time[example_index]

    return make_trace(x, y, truth=True, name="centorid pv")


def get_trace_all_pv_systems(pv: PV, example_index: int) -> List[go.Scatter]:
    """Produce plot of centroid pv system"""

    traces = []
    x = pv.time[example_index]
    n_pv_systems = pv.data.shape[2]

    for pv_system_index in range(1, n_pv_systems):
        y = pv.data[example_index, :, pv_system_index]

        truth = False
        name = f"pv system {pv_system_index}"

        traces.append(make_trace(x, y, truth=truth, name=name))

    centroid_trace = get_trace_centroid_pv(pv=pv, example_index=example_index)
    traces.append(centroid_trace)

    return traces


def get_traces_pv_intensity(pv: PV, example_index: int):
    """Get traces of pv intenisty map"""
    time = pv.time[example_index]
    x = pv.x_coords[example_index]
    y = pv.y_coords[example_index]

    n_pv_systems = pv.data.shape[2]

    traces = [go.Choroplethmapbox(colorscale="Viridis")]

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    for t_index in range(len(time)):
        z = pv.data[example_index, t_index, :]
        name = time[t_index].data

        trace = go.Scattermapbox(
            lat=lat,
            lon=lon,
            marker=dict(color=["Blue"] + ["Red"] * (n_pv_systems - 1), size=10 * z + 2),
            name=str(name),
        )
        traces.append(trace)

    return traces


def make_buttons() -> dict:
    """Make buttons Play dict"""
    return dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None]),
                                         dict(args =[[None], {"frame": {"duration": 0, "redraw": False},
                                                               "mode": "immediate",
                                                               "transition": {"duration": 0}}],
                                             label="Pause",
                                             method="animate")
                                         ])


def make_slider(labels: List[str]) -> dict:
    """ Make slider for animation"""
    sliders = [dict(steps=[dict(method='animate',
                                args=[[f'frame{k}'],
                                      dict(mode='immediate',
                                           frame=dict(duration=600, redraw=True),
                                           transition=dict(duration=200)
                                           )
                                      ],
                                label=f'{labels[k]}'
                                ) for k in range(0, len(labels))],
                    transition=dict(duration=100),
                    x=0,
                    y=0,
                    currentvalue=dict(font=dict(size=12), visible=True, xanchor='center'),
                    len=1.0)
               ]
    return sliders


def make_fig_of_animation_from_frames(traces, pv, example_index):
    """Make animated fig form traces"""

    frames = []
    for i, trace in enumerate(traces[1:]):
        frames.append(go.Frame(data=trace,name=f'frame{i}'))

    # make slider
    labels = [pd.to_datetime(time.data) for time in  pv.time[example_index]]
    sliders = make_slider(labels=labels)

    x = pv.x_coords[example_index].mean()
    y = pv.y_coords[example_index].mean()

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

    traces_pv_intensity_in_time = get_trace_all_pv_systems(pv=pv, example_index=example_index)

    traces_pv_intensity_map = get_traces_pv_intensity(pv=pv, example_index=example_index)

    x = pv.x_coords[example_index].mean()
    y = pv.y_coords[example_index].mean()

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
        frames.append(dict(data=[trace] + static_traces, traces=list(range(n_traces)), name=f'frame{i}'))

    # make slider
    labels = [pd.to_datetime(time.data) for time in  pv.time[example_index]]
    sliders = make_slider(labels=labels)

    fig.update(frames=frames)
    fig.update_layout(updatemenus=[make_buttons()])

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=8, mapbox_center={"lat": lat, "lon": lon}
    )

    fig.update_layout(sliders=sliders)


    return fig
