""" PLotting script for real nwp data """
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_utils.visualization.data_sources.plot_nwp import make_traces_nwp_one_channel_one_time
from nowcasting_utils.visualization.data_sources.plot_pv import get_trace_pv_intensity_one_time_step
from plotly.subplots import make_subplots

from nowcasting_dataset.geospatial import osgb_to_lat_lon


# load batch
tmp_path = "./temp"
src_path = 's3://solar-pv-nowcasting-data/prepared_ML_training_data/v16/train'
batch_idx = 4
example_index=0 # is good Newcastle
example_index=12 # is good London
example_index=20
time_index = 10

# **********************
# Optional download files from aws
# from nowcasting_dataset.filesystem.utils import download_to_local
# from nowcasting_dataset.utils import get_netcdf_filename
# from nowcasting_dataset.consts import SPATIAL_AND_TEMPORAL_LOCATIONS_OF_EACH_EXAMPLE_FILENAME
#
# data_sources_names = ['gsp','pv','satellite','nwp', 'hrvsatellite','topographic','opticalflow', 'sun']
# for data_source in data_sources_names:
#     data_source_and_filename = f"{data_source}/{get_netcdf_filename(batch_idx)}"
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
#
# **********************

batch = Batch.load_netcdf(tmp_path, batch_idx)


gsp = batch.gsp
pv = batch.pv
satellite = batch.satellite
nwp = batch.nwp

# loop over examples
for example_index in range(5,6):

    x = nwp.x[example_index].mean()
    y = nwp.y[example_index].mean()

    lat, lon = osgb_to_lat_lon(x=x,y=y)

    # make nwp sub plots
    traces = []
    for time_index in [1,2,3]:
        trace = make_traces_nwp_one_channel_one_time(nwp=nwp,example_index=example_index,channel_index=2,time_index=time_index)
        traces.append(trace)

    # make pv sub plots
    traces_pv = []
    for time_index in [5,17,29]:
        trace = get_trace_pv_intensity_one_time_step(pv=pv,example_index=example_index,t_index=time_index)
        traces_pv.append(trace)

    # make figure
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=('11:00', "12:00","13:00"),
        specs=[
            [{"type": "choroplethmapbox"}, {"type": "choroplethmapbox"}, {"type": "choroplethmapbox"}],
        ],
    )
    for i in range(len(traces)):
        fig.add_trace(traces[i],1,i+1)

    for i in range(len(traces_pv)):
        fig.add_trace(traces_pv[i],1,i+1)

    # fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=7.2, mapbox_center={"lat": lat, "lon": lon})
    # fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    mapbox = dict(style="carto-positron", center=dict(lat=lat, lon=lon), zoom=7.2)

    layout_dict = {f"mapbox{i}": mapbox for i in range(1, len(traces) + 1)}
    fig.update_layout(layout_dict)
    fig.show(renderer="browser")




