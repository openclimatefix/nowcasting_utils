""" Test pv plot functions """
import os

import plotly.graph_objects as go
from nowcasting_dataset.data_sources.fake.batch import pv_fake
from nowcasting_dataset.geospatial import osgb_to_lat_lon

from nowcasting_utils.visualization.data_sources.plot_pv import (
    get_fig_pv_combined,
    get_trace_all_pv_systems,
    get_trace_centroid_pv,
    get_traces_pv_intensity,
    make_fig_of_animation_from_frames,
)


def test_get_trace_centroid_pv(configuration):
    """Test 'get_trace_centroid_pv' function"""
    pv = pv_fake(configuration=configuration)

    trace = get_trace_centroid_pv(pv=pv, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    fig.add_trace(trace)
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_trace_all_pv_systems(configuration):
    """Test 'get_trace_all_pv_systems' function"""
    pv = pv_fake(configuration=configuration)

    traces = get_trace_all_pv_systems(pv=pv, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_traces_pv_intensity(configuration):
    """Test 'get_traces_pv_intensity' function"""
    pv = pv_fake(configuration=configuration)

    example_index = 1
    traces = get_traces_pv_intensity(pv=pv, example_index=1)

    x = pv.x_coords[example_index].mean()
    y = pv.y_coords[example_index].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    fig = go.Figure()
    for trace in traces[0:2]:
        fig.add_trace(trace)

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=8, mapbox_center={"lat": lat, "lon": lon}
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_traces_pv_intensity_and_animate(configuration):
    """Test 'make_fig_of_animation_from_frames' function"""
    pv = pv_fake(configuration=configuration)

    traces = get_traces_pv_intensity(pv=pv, example_index=1)

    fig = make_fig_of_animation_from_frames(traces=traces, pv=pv, example_index=1)

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_fig_pv_combined(configuration):
    """Test 'get_fig_pv_combined' function"""
    pv = pv_fake(configuration=configuration)

    fig = get_fig_pv_combined(pv=pv, example_index=1)
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")

    fig.write_html("pv_plot.html")
