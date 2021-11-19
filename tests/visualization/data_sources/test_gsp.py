""" Test gsp plot functions """
import os

import plotly.graph_objects as go
from nowcasting_dataset.data_sources.fake import gsp_fake
from nowcasting_dataset.geospatial import osgb_to_lat_lon

from nowcasting_utils.visualization.data_sources.plot_gsp import (
    get_fig_gsp_combined,
    get_trace_all_gsps,
    get_trace_centroid_gsp,
    get_traces_gsp_intensity,
    make_fig_of_animation_from_frames,
)


def test_get_trace_centroid_gsp():
    """ Test get trace for center gsp"""
    gsp = gsp_fake(batch_size=2, seq_length_30=5, n_gsp_per_batch=32)

    trace = get_trace_centroid_gsp(gsp=gsp, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    fig.add_trace(trace)
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_trace_all_gps():
    """ Test get traces for all gsps"""
    gsp = gsp_fake(batch_size=2, seq_length_30=5, n_gsp_per_batch=32)

    traces = get_trace_all_gsps(gsp=gsp, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_traces_gsp_intensity():
    """ Test get traces for gsp intensity """
    gsp = gsp_fake(batch_size=2, seq_length_30=5, n_gsp_per_batch=32)

    example_index = 1
    traces = get_traces_gsp_intensity(gsp=gsp, example_index=1)

    x = gsp.x_coords[example_index].mean()
    y = gsp.y_coords[example_index].mean()

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


def test_get_traces_gsp_intensity_and_animate():
    """ Test to get traces for gsp intensity and make animation"""
    gsp = gsp_fake(batch_size=2, seq_length_30=5, n_gsp_per_batch=32)

    traces = get_traces_gsp_intensity(gsp=gsp, example_index=1)

    fig = make_fig_of_animation_from_frames(traces=traces, gsp=gsp, example_index=1)

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_fig_gsp_combined():
    """ Test gsp combined plot"""
    gsp = gsp_fake(batch_size=2, seq_length_30=5, n_gsp_per_batch=32)

    fig = get_fig_gsp_combined(gsp=gsp, example_index=1)

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")

    fig.write_html("pv_plot.html")
