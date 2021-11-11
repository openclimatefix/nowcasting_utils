from typing import List

from nowcasting_dataset.data_sources.pv.pv_data_source import PV
from nowcasting_utils.visualization.line import make_trace
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    """ Get traces of pv intenisty map """
    time = pv.time[example_index]
    x = pv.x_coords[example_index]
    y = pv.y_coords[example_index]

    n_pv_systems  =pv.data.shape[2]

    traces = []
    for t_index in range(len(time)):
        z = pv.data[example_index, t_index, :]
        name = time[t_index].data
        traces.append(
            make_trace(x, y, truth=False, mode="markers", marker_size=10 * z + 1, name=str(name),
                       color = ['Blue'] + ['Red']*(n_pv_systems-1))
        )

    return traces


def make_buttons() -> dict:
    """ make buttons Play dict """
    return dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])


def make_fig_of_animation_from_frames(traces):
    """ Make animated fig form traces """
    frames = [go.Frame(data=trace) for trace in traces]
    fig = go.Figure(
        data=traces[0],
        layout=go.Layout(
            title="Start Title",
            # updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])],
        ),
        frames=frames,
    )
    fig.update_layout(updatemenus=[make_buttons()])

    return fig


def get_fig_pv_combined(pv: PV, example_index: int):
    """
    Create a combined plot

    1. Plot the pv intensity in time
    2. Plot the pv intensity with coords and animate in time
    """

    traces_pv_intensity_in_time = get_trace_all_pv_systems(pv=pv, example_index=example_index)

    traces_pv_intensity_map = get_traces_pv_intensity(pv=pv, example_index=example_index)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Map', 'Time Series'))

    # add first animation plot
    fig.add_trace(traces_pv_intensity_map[0], row=1, col=1)

    # add all time series plots
    for trace in traces_pv_intensity_in_time:
        fig.add_trace(trace, row=1, col=2)

    n_traces = len(fig.data)

    frames = []
    static_traces = list(fig.data[1:])
    for trace in traces_pv_intensity_map:
        frames.append(dict(
            data=[trace] + static_traces,
            traces=list(range(n_traces))
        ))

    fig.update(frames=frames)
    fig.update_layout(updatemenus=[make_buttons()])

    return fig
