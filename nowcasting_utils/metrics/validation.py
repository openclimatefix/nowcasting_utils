""" Functions to save validation results to logger/csv """
import logging
from typing import List, Optional

from datetime import timedelta

import pandas as pd
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

_log = logging.getLogger(__name__)


def make_validation_results(
    predictions_mw, truths_mw, gsp_ids: List[int], t0_datetimes_utc: pd.DatetimeIndex, batch_idx: Optional[int] = None, forecast_sample_period: timedelta = timedelta(minutes=30)
) -> pd.DataFrame:
    """
    Make validations results.

    The following columns are made:

    - t0_datetime_utc
    - gsp_id

    - prediction_{i} where i in range(0,n_forecast_steps)
    - truth_{i} where i in range(0,n_forecast_steps)
    - batch_index (optional)
    - example_index (optional)

    return: Dataframe of predictions and truths
    """

    assert predictions_mw.shape == truths_mw.shape

    results_per_forecast_horizon = []

    for i in range(predictions_mw.shape[1]):
        predictions_mw_df = pd.DataFrame(predictions_mw[:, i], columns=['forecast_gsp_pv_outturn_mw'])
        predictions_mw_df['actual_gsp_pv_outturn_mw'] = truths_mw[:, i]
        predictions_mw_df["target_datetime_utc"] = t0_datetimes_utc + (i+1)*forecast_sample_period
        predictions_mw_df["gsp_id"] = gsp_ids
        predictions_mw_df["t0_datetime_utc"] = t0_datetimes_utc

        results_per_forecast_horizon.append(predictions_mw_df)

    # join truths and predictions for each forecast horizon
    results = pd.concat(results_per_forecast_horizon)

    # join truths and predictions
    results.index.name = "example_index"

    # add batch index
    if batch_idx is not None:
        results["batch_index"] = batch_idx

    return results


def save_validation_results_to_logger(
    results_dfs: List[pd.DataFrame],
    results_file_name: str,
    current_epoch: int,
    logger: Optional[NeptuneLogger] = None,
):
    """
    Save validation results to logger
    """

    _log.info("Saving results of validation to logger")
    if logger is None:
        _log.debug("logger is not set, so not saving validation results")
    else:
        _log.info("Saving results of validation to logger")

        # join all validation step results together
        results_df = pd.concat(results_dfs)
        results_df.reset_index(inplace=True)

        # save to csv file
        name_csv = f"{results_file_name}_{current_epoch}.csv"
        results_df.to_csv(name_csv)

        # upload csv to neptune
        try:
            logger.experiment[-1][f"validation/results/epoch_{current_epoch}"].upload(name_csv)
        except Exception as e:
            _log.debug(e)
            _log.debug("Could not save validation results to logger")
