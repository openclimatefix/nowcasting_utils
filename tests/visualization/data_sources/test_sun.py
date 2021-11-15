from nowcasting_utils.visualization.data_sources.plot_sun import (
    get_elevation_and_azimuth_trace
)
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from nowcasting_dataset.data_sources.fake import (
    sun_fake,
)
import os
import plotly.graph_objects as go


def test_get_trace_centroid_pv():

    sun = sun_fake(batch_size=2, seq_length_5=19)

    traces = get_elevation_and_azimuth_trace(sun=sun, example_index=1)

    # here's if you need to plot the trace
    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")
