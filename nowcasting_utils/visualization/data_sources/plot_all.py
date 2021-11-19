""" Tests to plot satellite data """
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.geospatial import osgb_to_lat_lon

from nowcasting_utils.visualization.data_sources.plot_gsp import (
    get_trace_all_gsps,
    get_trace_gsp_intensity_one_time_step,
)
from nowcasting_utils.visualization.data_sources.plot_pv import (
    get_trace_all_pv_systems,
    get_trace_pv_intensity_one_time_step,
)
from nowcasting_utils.visualization.data_sources.plot_satellite import (
    make_traces_one_channel_one_time,
)
from nowcasting_utils.visualization.utils import make_buttons, make_slider


def make_trace_time_series_pv_and_gsp(batch: Batch, example_index: int):
    """
    Make traces of time series gsp and pv systems
    """

    pv = batch.pv
    gsp = batch.gsp

    traces_pv = get_trace_all_pv_systems(pv=pv, example_index=example_index, center_system=False)
    traces_gsp = get_trace_all_gsps(gsp=gsp, example_index=example_index)

    return traces_gsp + traces_pv


def make_fig_time_series_pv_and_gsp(batch: Batch, example_index: int):
    """
    Make figure of pv and gsp time series data
    """

    traces = make_trace_time_series_pv_and_gsp(batch=batch, example_index=example_index)

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    fig.update_layout(
        title="GSP and PV time series plot",
    )

    return fig


def get_time_index(times: xr.DataArray, time_value) -> int:
    """Get the time index from a time value"""
    previous_and_equal_mask = times <= time_value
    previous_and_equal_times_index = times[previous_and_equal_mask].time_index

    if len(previous_and_equal_times_index) == 0:
        raise Exception(f"No time index for {time_value}")

    return int(previous_and_equal_times_index[-1])


def make_satellite_gsp_pv_map_one_time_value(
    batch: Batch, example_index: int, satellite_channel_index, time_value
):
    """Make plot of satellite, gps and pv for one time step"""

    pv = batch.pv
    gsp = batch.gsp
    satellite = batch.satellite

    pv_time_index = get_time_index(pv.time[example_index], time_value)
    gsp_time_index = get_time_index(gsp.time[example_index], time_value)
    satellite_time_index = get_time_index(satellite.time[example_index], time_value)

    trace_pv = get_trace_pv_intensity_one_time_step(
        pv=pv, example_index=example_index, t_index=pv_time_index
    )
    trace_gsp = get_trace_gsp_intensity_one_time_step(
        gsp=gsp, example_index=example_index, t_index=gsp_time_index
    )
    trace_satellite = make_traces_one_channel_one_time(
        satellite=satellite,
        example_index=example_index,
        channel_index=satellite_channel_index,
        time_index=satellite_time_index,
    )

    # return traces_gsp + traces_pv + [trace_satellite]
    return [trace_pv, trace_gsp, trace_satellite]


def make_satellite_gsp_pv_map(batch: Batch, example_index: int, satellite_channel_index: int):
    """Make a animation of the satellite, gsp and the pv data"""
    trace_times = []
    times = batch.satellite.time[example_index]
    pv = batch.pv

    for time in times:
        trace_times.append(
            make_satellite_gsp_pv_map_one_time_value(
                batch=batch,
                example_index=example_index,
                satellite_channel_index=satellite_channel_index,
                time_value=time,
            )
        )

    frames = []
    for i, traces in enumerate(trace_times):
        frames.append(go.Frame(data=traces, name=f"frame{i+1}"))

    # make slider
    labels = [pd.to_datetime(time.data) for time in times]
    sliders = make_slider(labels=labels)

    x = pv.x_coords[example_index][pv.x_coords[example_index] != 0].mean()
    y = pv.y_coords[example_index][pv.y_coords[example_index] != 0].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    fig = go.Figure(
        data=trace_times[0],
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
