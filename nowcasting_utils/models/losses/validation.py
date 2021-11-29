import pandas as pd
from typing import Optional


def make_validation_results(predictions, truths, gsp_ids, t0_datetimes_utc,
                            batch_idx: Optional[int] = None) -> pd.DataFrame:
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
