""" PLotting script for real data """
from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.filesystem.utils import download_to_local
from nowcasting_dataset.utils import get_netcdf_filename

from nowcasting_utils.visualization.data_sources.plot_all import (
    make_fig_time_series_pv_and_gsp,
    make_satellite_gsp_pv_map,
    make_satellite_gsp_pv_map_still,
)
from nowcasting_utils.visualization.data_sources.plot_gsp import get_fig_gsp_combined
from nowcasting_utils.visualization.data_sources.plot_pv import get_fig_pv_combined
from nowcasting_utils.visualization.data_sources.plot_satellite import make_animation_all_channels


# load batch
tmp_path = "./temp"
src_path = "s3://solar-pv-nowcasting-data/prepared_ML_training_data/v16/train"
batch_idx = 1
example_index = 16

# download files from aws
# from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
# from nowcasting_dataset.filesystem.utils import download_to_local
# from nowcasting_dataset.utils import get_netcdf_filename
# data_sources_names = ['gsp','pv','satellite','nwp', 'hrvsatellite','topographic','opticalflow', 'sun']
# for data_source in data_sources_names:
#     data_source_and_filexname = f"{data_source}/{get_netcdf_filename(batch_idx)}"
#     download_to_local(
#         remote_filename=f"{src_path}/{data_source_and_filename}",
#         local_filename=f"{tmp_path}/{data_source_and_filename}",
#     )
#
#     # download locations file
# download_to_local(
#     remote_filename=f"{src_path}/"
#                     f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}",
#     local_filename=f"{tmp_path}/"
#                    f"{SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME}",
# )

batch = Batch.load_netcdf(tmp_path, batch_idx)


# setup
time_index = 10

gsp = batch.gsp
pv = batch.pv
satellite = batch.satellite


def plot_get_fig_gsp_combined():
    """Plot gsp"""

    fig = get_fig_gsp_combined(gsp=gsp, example_index=example_index)
    fig.show(renderer="browser")
    fig.write_html("./gsp_plot.html")


def plot_get_fig_pv_combined(example_index=example_index):
    """Plot pv"""

    fig = get_fig_pv_combined(pv=pv, example_index=example_index)
    fig.show(renderer="browser")
    fig.write_html("./pv_plot.html")


def plot_make_animation_all_channels():
    """Plot satellite"""

    fig = make_animation_all_channels(satellite=satellite, example_index=example_index)
    fig.show(renderer="browser")
    fig.write_html("./satellite_plot.html")


def plot_make_fig_time_series_pv_and_gsp(batch, example_index):
    """Plot pv and gsp"""
    fig = make_fig_time_series_pv_and_gsp(batch=batch, example_index=example_index)
    fig.show(renderer="browser")


def plot_make_satellite_gsp_pv_map(batch):
    """Animate satellite, pv and gsp"""
    fig = make_satellite_gsp_pv_map(
        batch=batch, example_index=example_index, satellite_channel_index=7
    )

    fig.show(renderer="browser")
    fig.write_html("./all_plot.html")


def plot_make_satellite_gsp_pv_map_still(batch, example_index):
    """Animate satellite, pv and gsp"""
    fig = make_satellite_gsp_pv_map_still(
        batch=batch, example_index=example_index, satellite_channel_index=0
    )

    fig.show(renderer="browser")
    fig.write_html("./all_plot_still.html")


plot_get_fig_gsp_combined()
plot_get_fig_pv_combined()
plot_make_animation_all_channels()
plot_make_fig_time_series_pv_and_gsp(batch)
plot_make_satellite_gsp_pv_map(batch)
for i in range(8, 10):
    plot_make_fig_time_series_pv_and_gsp(batch, i)
    plot_make_satellite_gsp_pv_map_still(batch, i)
