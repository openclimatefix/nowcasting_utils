""" PLotting script for real data """
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_utils.visualization.data_sources.plot_all import (

    make_fig_time_series_pv_and_gsp,
    make_satellite_gsp_pv_map_one_time_value,
    make_satellite_gsp_pv_map,
)

import plotly.graph_objects as go
from nowcasting_dataset.geospatial import osgb_to_lat_lon

from nowcasting_utils.visualization.data_sources.plot_gsp import get_fig_gsp_combined

from nowcasting_utils.visualization.data_sources.plot_gsp import (
    get_fig_gsp_combined,
)
from nowcasting_utils.visualization.data_sources.plot_pv import get_fig_pv_combined
from nowcasting_utils.visualization.data_sources.plot_satellite import make_animation_all_channels

# load batch
file_path = "/Users/peterdudfield/Documents/Github/nowcasting_utils/train"
batch = Batch.load_netcdf(file_path, 0)

# normalize data - should change this
gsp = batch.gsp
pv = batch.pv
satellite = batch.satellite

gsp.__setitem__("power_mw", (gsp.power_mw / gsp.capacity_mwp).fillna(0))
pv.__setitem__("power_mw", pv.power_normalized.fillna(0))

batch.pv = pv
batch.gsp = gsp
batch.satellite = satellite


# setup
example_index = 0
time_index = 6


def plot_get_fig_gsp_combined():
    """ Plot gsp """

    fig = get_fig_gsp_combined(gsp=gsp, example_index=1)
    fig.show(renderer="browser")
    fig.write_html("pv_plot.html")


def plot_get_fig_pv_combined():
    """ Plot pv """

    fig = get_fig_pv_combined(pv=pv, example_index=1)
    fig.show(renderer="browser")
    fig.write_html("pv_plot.html")


def plot_make_animation_all_channels():
    """ Plot satellite """

    fig = make_animation_all_channels(satellite=satellite, example_index=0)
    fig.show(renderer="browser")


def plot_make_fig_time_series_pv_and_gsp(batch):
    """ Plot pv and gsp """
    fig = make_fig_time_series_pv_and_gsp(batch=batch, example_index=18)
    fig.show(renderer="browser")


def plot_make_satellite_gsp_pv_map_one_time_step(batch):
    """ Plot satellite, gsp and pv """
    time_value = satellite.time[example_index, time_index]
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

    fig.show(renderer="browser")


def plot_make_satellite_gsp_pv_map(batch):
    """ Animate satellite, pv and gsp """
    fig = make_satellite_gsp_pv_map(batch=batch, example_index=15, satellite_channel_index=7)

    fig.show(renderer="browser")


plot_get_fig_gsp_combined()
plot_get_fig_pv_combined()
plot_make_animation_all_channels()
plot_make_fig_time_series_pv_and_gsp(batch)
plot_make_satellite_gsp_pv_map(batch)
plot_make_satellite_gsp_pv_map_one_time_step(batch)
