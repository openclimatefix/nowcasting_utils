""" Functions to save validation results to logger/csv """
import logging
from neptune.new.integrations.pytorch_lightning import NeptuneLogger
from typing import List, Optional

import pandas as pd

_log = logging.getLogger(__name__)


def make_validation_results(
    predictions, truths, gsp_ids, t0_datetimes_utc, batch_idx: Optional[int] = None
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

    predictions = pd.DataFrame(
        predictions, columns=[f"prediction_{i}" for i in range(predictions.shape[1])]
    )
    truths = pd.DataFrame(truths, columns=[f"truth_{i}" for i in range(truths.shape[1])])

    # join truths and predictions
    results = pd.concat([predictions, truths], axis=1, join="inner")
    results.index.name = "example_index"

    # add metadata
    results["gsp_id"] = gsp_ids
    results["t0_datetime_utc"] = t0_datetimes_utc
    if batch_idx is not None:
        results["batch_index"] = batch_idx

    return results


def save_validation_results_to_logger(
    results_dfs: List[pd.DataFrame], results_file_name: str, current_epoch: int, logger: Optional[NeptuneLogger] = None
):
    """
    Save validation results to logger
    """

    _log.info("Saving results of validation to logger")
    if logger is None:
        _log.debug('logger is not set, so not saving validation results')
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
