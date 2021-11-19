""" Test all plot functions """
import os

import plotly.graph_objects as go
from nowcasting_dataset.geospatial import osgb_to_lat_lon

from nowcasting_utils.visualization.data_sources.plot_all import (
    make_fig_time_series_pv_and_gsp,
    make_satellite_gsp_pv_map,
    make_satellite_gsp_pv_map_one_time_value,
)


def test_make_fig_time_series_pv_and_gsp(batch):
    """Test 'make_fig_time_series_pv_and_gsp' function"""

    fig = make_fig_time_series_pv_and_gsp(batch=batch, example_index=1)

    # here's if you need to plot the trace
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_make_satellite_gsp_pv_map_one_time_step(batch):
    """Make plot of satellite, gsp and pv"""
    example_index = 0
    time_index = 6
    time_value = batch.satellite.time[example_index, time_index]
    traces = make_satellite_gsp_pv_map_one_time_value(
        batch=batch, example_index=example_index, satellite_channel_index=0, time_value=time_value
    )

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)

    x_osgb = float(batch.gsp.y_coords[example_index, 0])
    y_osgb = float(batch.gsp.x_coords[example_index, 0])

    lat, lon = osgb_to_lat_lon(x=x_osgb, y=y_osgb)

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=7, mapbox_center={"lat": lat, "lon": lon}
    )

    # here's if you need to plot the trace
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_make_satellite_gsp_pv_map(batch):
    """Test make animation of satelite, gsp and pv"""

    fig = make_satellite_gsp_pv_map(batch=batch, example_index=1, satellite_channel_index=7)

    # here's if you need to plot the trace
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")

    # fig.write_html("batch_all_plot.html")
