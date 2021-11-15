""" General functions for plotting PV data """

import plotly.graph_objects as go
from nowcasting_dataset.data_sources.sun.sun_model import Sun
from typing import List

from nowcasting_utils.visualization.line import make_trace


def get_elevation_and_azimuth_trace(sun: Sun, example_index: int) -> List[go.Scatter]:
    """Produce plot of centroid pv system"""

    y1 = sun.elevation[example_index]
    y2 = sun.azimuth[example_index]
    x = sun.time[example_index]

    trace_elevation = make_trace(x, y1, truth=True, name="elevation")
    trace_azimuth = make_trace(x, y2, truth=True, name="azimuth")

    return [trace_elevation, trace_azimuth]
