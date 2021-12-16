""" Plot of specific gsp ids """
import json
import pytz
from datetime import datetime

import plotly.graph_objects as go
import xarray as xr
from nowcasting_dataset.data_sources.gsp.eso import (
    get_gsp_metadata_from_eso,
    get_gsp_shape_from_eso,
)
from pvlive_api import PVLive

WGS84_CRS = "EPSG:4326"

shapes_gdp = get_gsp_shape_from_eso()
gsp_metadata = get_gsp_metadata_from_eso()

missing_gsp_ids = [
    18,
    25,
    34,
    41,
    47,
    50,
    76,
    89,
    112,
    114,
    160,
    164,
    175,
    180,
    185,
    203,
    206,
    215,
    217,
    222,
    258,
    260,
    270,
    278,
    313,
    314,
    316,
    317,
    318,
    331,
    333,
]

# missing_gsp_ids = [160, 164, 175, 180, 206, 217]

shapes_gdp = shapes_gdp.to_crs(WGS84_CRS)
gsp_metadata = gsp_metadata.to_crs(WGS84_CRS)

gsp_metadata = gsp_metadata[gsp_metadata["gsp_id"].isin(missing_gsp_ids)]
shapes_gdp = shapes_gdp[shapes_gdp["RegionID"].isin(missing_gsp_ids)]
gsp_metadata["Amount"] = 0.0
shapes_gdp["Amount"] = 0.0
#
# gsp_ids = sorted(gsp_metadata['gsp_id'].unique())
# empty_gsp = [x for x in range(1, 339) if x not in gsp_ids]
# print(f'GSP with no data are {empty_gsp}')

gsp_data_to_plot = gsp_metadata

shapes_dict = json.loads(gsp_data_to_plot.to_json())

# plot it
fig = go.Figure()
fig.add_trace(
    go.Choroplethmapbox(
        geojson=shapes_dict,
        locations=gsp_data_to_plot.index,
        z=gsp_data_to_plot.Amount,
        colorscale="Viridis",
    )
)

fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=4, mapbox_center={"lat": 55, "lon": 0})
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show(renderer="browser")
fig.write_html("gsp_missing.html")

# ## load raw data
start_dt = datetime.fromisoformat("2020-01-01")
end_dt = datetime.fromisoformat("2021-09-01")
end_dt = datetime.fromisoformat("2020-01-02")

filename = "gs://solar-pv-nowcasting-data/PV/GSP/v2/pv_gsp.zarr"

gsp_power = xr.open_dataset(filename, engine="zarr")
gsp_power = gsp_power.generation_mw
gsp_power = gsp_power.sel(datetime_gmt=slice(start_dt, end_dt))
gsp_power["gsp_id"] = gsp_power.gsp_id.astype(int)


maximum_gsp = gsp_power.max(dim="datetime_gmt")
gsp_ids_with_nans = maximum_gsp[maximum_gsp.isnull()].gsp_id.values


nans = [x for x in missing_gsp_ids if x in gsp_ids_with_nans]
no_nans = [x for x in missing_gsp_ids if x not in gsp_ids_with_nans]


# load data from pv live
gsp_id = 34
start_dt = datetime.fromisoformat("2020-07-01").replace(tzinfo=pytz.utc)
end_dt = datetime.fromisoformat("2021-09-01").replace(tzinfo=pytz.utc)
end_dt = datetime.fromisoformat("2020-07-02").replace(tzinfo=pytz.utc)

for gsp_id in missing_gsp_ids:
    pvl = PVLive()
    pv_df = pvl.between(
        start=start_dt,
        end=end_dt,
        entity_type="gsp",
        entity_id=gsp_id,
        extra_fields="installedcapacity_mwp",
        dataframe=True,
    )

    print(pv_df.max())
