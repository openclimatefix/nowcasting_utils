""" PLotting script for real data """
from nowcasting_dataset.dataset.batch import Batch
from nowcasting_utils.visualization.data_sources.plot_satellite import make_traces_one_channel_one_time
from nowcasting_utils.visualization.data_sources.plot_pv import get_trace_pv_intensity_one_time_step
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nowcasting_dataset.geospatial import osgb_to_lat_lon

# load batch
tmp_path = "./temp"
src_path = 's3://solar-pv-nowcasting-data/prepared_ML_training_data/v16/train'
batch_idx = 1
example_index=0 # is good Newcastle
example_index=12 # is good London
example_index=20
time_index = 10

batch = Batch.load_netcdf(tmp_path, batch_idx)


gsp = batch.gsp
pv = batch.pv
satellite = batch.satellite

for example_index in range(11,13):

    x = satellite.x[example_index].mean()
    y = satellite.y[example_index].mean()

    lat, lon = osgb_to_lat_lon(x=x,y=y)

    traces = []
    for time_index in [0,15,30]:
        trace = make_traces_one_channel_one_time(satellite=satellite,example_index=example_index,channel_index=8,time_index=time_index)
        traces.append(trace)

    traces_pv = []
    for time_index in [0,15,30]:
        trace = get_trace_pv_intensity_one_time_step(pv=pv,example_index=example_index,t_index=time_index)
        traces_pv.append(trace)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=('13:00', "14:00","15:00"),
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




