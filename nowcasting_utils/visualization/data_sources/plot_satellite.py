import plotly.graph_objects as go
from nowcasting_dataset.data_sources.satellite.satellite_data_source import Satellite
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from nowcasting_utils.visualization.utils import make_slider, make_buttons
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots


def make_traces_one_channel_one_time(
    satellite: Satellite, example_index: int, channel_index: int, time_index: int
):
    """
    Make one trace for one channel and one time

    Args:
        satellite:
        example_index:
        channel_index:
        time_index:

    Returns:

    """
    z = satellite.data[example_index, time_index, :, :, channel_index].values

    x = satellite.x[example_index]
    y = satellite.y[example_index]

    lat, lon = osgb_to_lat_lon(x, y)

    # lets un ravel the z object
    z = z.ravel()
    # now for each z object we need to make a lat and lon
    # if lat = [53,54], lon = [0,1]
    # then we want to change it to
    # lat = [53,54,43,54] and  lon = [0,0,1,1]
    lat = np.tile(lat, len(y))
    lon = np.repeat(lon, len(x))

    return go.Densitymapbox(
        z=z, lat=lat, lon=lon, colorscale="Viridis", opacity=0.5, zmax=1000, zmin=0
    )


def make_traces_one_channel(satellite: Satellite, example_index: int, channel_index: int):
    """Make traces for one channel"""

    time = satellite.time[example_index]

    traces = []
    for time_index in range(len(time)):
        traces.append(
            make_traces_one_channel_one_time(
                satellite=satellite,
                example_index=example_index,
                channel_index=channel_index,
                time_index=time_index,
            )
        )

    return traces


def make_traces_one_timestep(satellite: Satellite, example_index: int, time_index: int):
    """Make traces for one channel"""

    channels = satellite.channels

    traces = []
    for channel_index in range(len(channels)):
        traces.append(
            make_traces_one_channel_one_time(
                satellite=satellite,
                example_index=example_index,
                channel_index=channel_index,
                time_index=time_index,
            )
        )

    return traces


def make_animation_one_channels(satellite: Satellite, example_index: int, channel_index: int):

    traces = make_traces_one_channel(
        satellite=satellite, example_index=example_index, channel_index=0
    )

    fig = go.Figure(
        data=traces[0],
        layout=go.Layout(
            title="Start Title",
        ),
    )

    frames = []
    for i, trace in enumerate(traces):
        frames.append(go.Frame(data=trace, name=f"frame{i + 1}"))

    fig.update(frames=frames)

    x = satellite.x[example_index].mean()
    y = satellite.y[example_index].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=7, mapbox_center={"lat": lat, "lon": lon}
    )

    fig.update_layout(updatemenus=[make_buttons()])

    return fig


def make_animation_all_channels(satellite: Satellite, example_index: int):
    """
    Make animation of all channels

    An animation is made over time. Subplots show the different satellite channels

    Args:
        satellite: satellite data
        example_index: which example to use

    Returns: plotly figure
    """
    time = satellite.time[example_index]

    # collect all the traces
    traces_all = []
    for i in range(len(time)):
        traces = make_traces_one_timestep(
            satellite=satellite, example_index=example_index, time_index=i
        )
        traces_all.append(traces)

    # make subplot
    n_rows = int(np.floor(len(satellite.channels) ** 0.5))
    n_cols = int(np.ceil(len(satellite.channels) ** 0.5))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        # subplot_titles= satellite.channels,
        specs=[
            [{"type": "choroplethmapbox"}] * n_rows * n_cols,
        ],
    )

    # add the first timestemp to the figure
    for i, trace in enumerate(traces_all[0]):
        row = i % n_rows + 1
        col = i // n_rows + 1
        print(row, col)
        fig.add_trace(trace, row, col)

    # add frames
    frames = []
    for i, traces in enumerate(traces_all):
        frames.append(dict(data=traces, traces=list(range(len(traces))), name=f"frame{i+1}"))

    fig.update(frames=frames)
    fig.update_layout(updatemenus=[make_buttons()])

    # make slider
    labels = [pd.to_datetime(time.data) for time in satellite.time[example_index]]
    sliders = make_slider(labels=labels)
    fig.update_layout(sliders=sliders)

    # sort out map layout
    x = satellite.x[example_index].mean()
    y = satellite.y[example_index].mean()
    lat, lon = osgb_to_lat_lon(x=x, y=y)

    mapbox = dict(style="carto-positron", center=dict(lat=lat, lon=lon), zoom=7)

    fig.update_layout(
        mapbox1=mapbox,
        mapbox2=mapbox,
    )

    return fig
