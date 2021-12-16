""" Tests to plot satellite data """
import numpy as np
import plotly.graph_objects as go
from nowcasting_dataset.data_sources.nwp.nwp_data_source import NWP
from nowcasting_dataset.geospatial import osgb_to_lat_lon


from nowcasting_dataloader.data_sources.nwp.nwp_model import NWP_MEAN, NWP_STD


def make_traces_nwp_one_channel_one_time(
    nwp: NWP, example_index: int, channel_index: int, time_index: int
):
    """
    Make one trace for one channel and one time

    Args:
        nwp: nwp data
        example_index: which example to use
        channel_index: which channel to ise
        time_index: which time to use

    Returns: plotly trace

    """
    z = nwp.data[example_index, channel_index, time_index, :, :]
    z = z - list(NWP_MEAN.values())[channel_index]
    z = z / list(NWP_STD.values())[channel_index]
    print(z)
    # z = -z

    z = z.transpose("y_index", "x_index").values

    x = nwp.x[example_index]
    y = nwp.y[example_index]

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    # lets un ravel the z object
    z = z.ravel()
    # now for each z object we need to make a lat and lon
    # if lat = [53,54], lon = [0,1]
    # then we want to change it to
    # lat = [53,54,43,54] and  lon = [0,0,1,1]
    lat = np.repeat(lat, len(y))
    lon = np.tile(lon, len(x))

    return go.Densitymapbox(
        z=z,
        lat=lat,
        lon=lon,
        colorscale=[[0, "white"], [0.9, "white"], [1, "blue"]],
        opacity=0.3,
        zmax=10,
        zmin=-2,
    )
