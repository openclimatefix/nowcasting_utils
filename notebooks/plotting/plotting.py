""" PLotting script for real data """
from nowcasting_dataset.dataset.batch import Batch

from nowcasting_utils.visualization.data_sources.plot_all import (
    make_fig_time_series_pv_and_gsp,
    make_satellite_gsp_pv_map,
)
from nowcasting_utils.visualization.data_sources.plot_gsp import get_fig_gsp_combined
from nowcasting_utils.visualization.data_sources.plot_pv import get_fig_pv_combined
from nowcasting_utils.visualization.data_sources.plot_satellite import make_animation_all_channels

# load batch
file_path = "./../train"
batch = Batch.load_netcdf(file_path, 0)


# setup
example_index = 0
time_index = 6

gsp = batch.gsp
pv = batch.pv
satellite = batch.satellite


def plot_get_fig_gsp_combined():
    """Plot gsp"""

    fig = get_fig_gsp_combined(gsp=gsp, example_index=1)
    fig.show(renderer="browser")
    fig.write_html("./gsp_plot.html")


def plot_get_fig_pv_combined():
    """Plot pv"""

    fig = get_fig_pv_combined(pv=pv, example_index=1)
    fig.show(renderer="browser")
    fig.write_html("./pv_plot.html")


def plot_make_animation_all_channels():
    """Plot satellite"""

    fig = make_animation_all_channels(satellite=satellite, example_index=0)
    fig.show(renderer="browser")
    fig.write_html("./satellite_plot.html")


def plot_make_fig_time_series_pv_and_gsp(batch):
    """Plot pv and gsp"""
    fig = make_fig_time_series_pv_and_gsp(batch=batch, example_index=18)
    fig.show(renderer="browser")


def plot_make_satellite_gsp_pv_map(batch):
    """Animate satellite, pv and gsp"""
    fig = make_satellite_gsp_pv_map(batch=batch, example_index=15, satellite_channel_index=7)

    fig.show(renderer="browser")
    fig.write_html("./all_plot.html")


plot_get_fig_gsp_combined()
plot_get_fig_pv_combined()
plot_make_animation_all_channels()
plot_make_fig_time_series_pv_and_gsp(batch)
plot_make_satellite_gsp_pv_map(batch)
