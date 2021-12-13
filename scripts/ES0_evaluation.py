""" Script to run ESO evaluation """
# imports
from pathlib import Path

import pandas as pd

from nowcasting_utils.metrics.evaluation import evaluation

# Output csv
ESO_PV_FORECASTS_OUTPUT_FILE = Path(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/other_organisations_pv_forecasts/"
    "National_Grid_ESO/CSV/testset_v16.csv"
)

# run evaluation for ESO forecast
model_name = "ESO_forecast_v16"

results_df = pd.read_csv(ESO_PV_FORECASTS_OUTPUT_FILE)
print(results_df)

evaluation(results_df, model_name, show_fig=False)
