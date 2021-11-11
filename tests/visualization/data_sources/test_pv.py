from nowcasting_utils.visualization.data_sources.plot_pv import (
    get_trace_centroid_pv,
    get_trace_all_pv_systems,
    get_traces_pv_intensity,
    make_fig_of_animation_from_frames,
    get_fig_pv_combined
)
from nowcasting_dataset.data_sources.fake import (
    pv_fake,
)
import os
import plotly.graph_objects as go


def test_get_trace_centroid_pv():

    pv = pv_fake(batch_size=2, seq_length_5=5, n_pv_systems_per_batch=32)

    trace = get_trace_centroid_pv(pv=pv, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    fig.add_trace(trace)
    if 'CI' not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_trace_all_pv_systems():

    pv = pv_fake(batch_size=2, seq_length_5=5, n_pv_systems_per_batch=32)

    traces = get_trace_all_pv_systems(pv=pv, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    if 'CI' not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_traces_pv_intensity():

    pv = pv_fake(batch_size=2, seq_length_5=5, n_pv_systems_per_batch=32)

    traces = get_traces_pv_intensity(pv=pv, example_index=1)

    fig = make_fig_of_animation_from_frames(traces=traces)

    if 'CI' not in os.environ.keys():
        fig.show(renderer="browser")


def test_get_fig_pv_combined():
    pv = pv_fake(batch_size=2, seq_length_5=19, n_pv_systems_per_batch=8)

    fig = get_fig_pv_combined(pv=pv, example_index=1)
    if 'CI' not in os.environ.keys():
        fig.show(renderer="browser")


