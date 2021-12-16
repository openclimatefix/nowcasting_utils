# create a custom plot where only some gsp id are plotted

import pandas as pd
import xarray as xr
import fsspec
import io
import json
import plotly.graph_objects as go
from nowcasting_dataset.data_sources.gsp.eso import get_gsp_shape_from_eso, get_gsp_metadata_from_eso

# gsp ids to plot
gsp_id =[261,170,332,280,228,255,162,158]

# PV files
pv_metadata_fileanme = "gs://solar-pv-nowcasting-data/PV/Passive/ocf_formatted/v0/system_metadata.csv"
pv_filename = "gs://solar-pv-nowcasting-data/PV/Passive/ocf_formatted/v0/passiv.netcdf"

# load pv files
pv_metadata = pd.read_csv(pv_metadata_fileanme)
with fsspec.open(pv_filename, mode="rb") as file:
    file_bytes = file.read()

with io.BytesIO(file_bytes) as file:
    pv_power = xr.open_dataset(file, engine="h5netcdf")
    print(pv_power)

# join pv power and metadata together
ids = list(pv_power.keys())
ids = [int(x) for x in ids]
pv = pv_metadata[pv_metadata.ss_id.isin(ids)]

# # get shape of GSPs
gsp_shape = get_gsp_shape_from_eso().to_crs("EPSG:4326")
# set z axis
gsp_shape["Amount"] = 0

# optional to only plot some gsp ids
gsp_metadata = get_gsp_metadata_from_eso()
gsp_metadata = gsp_metadata[gsp_metadata['gsp_id'].isin(gsp_id)]
gsp_shape = gsp_shape[gsp_shape.RegionID.isin(gsp_metadata.region_id)]

# get dict shapes
shapes_dict = json.loads(gsp_shape.to_json())


# plot GSP shape
trace_gsp = go.Choroplethmapbox(
        geojson=shapes_dict, locations=gsp_shape.index, z=gsp_shape.Amount, colorscale="Viridis", marker=dict(opacity=0.5)
    )

# plot PVs
trace_pv = go.Scattermapbox(
        lat=pv.latitude,
        lon=pv.longitude,
        marker=dict(color="Red", size=3, sizemode="area"),
        name='PV',
        text=pv.system_id,
    )

# make figure
fig = go.Figure(
        data=[trace_gsp, trace_pv],
        layout=go.Layout(
            title="PV systems",
        ),
    )
fig.update_layout(
    mapbox_style="carto-positron", mapbox_zoom=4, mapbox_center={"lat": 54, "lon": -1}
)

fig.show(renderer='browser')