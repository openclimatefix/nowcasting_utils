""" Test saving validation results """
import numpy as np
import pandas as pd

from nowcasting_utils.metrics.validation import (
    make_validation_results,
    save_validation_results_to_logger,
)


def test_make_validation_results():
    """Test making the validation"""

    batch_size = 2
    forecast_length = 4

    predictions = np.random.random(size=(batch_size, forecast_length))
    capacity = np.random.random(size=(batch_size, forecast_length))
    truths = np.random.random(size=(batch_size, forecast_length))
    t0_datetimes_utc = pd.to_datetime(np.random.randint(low=0, high=1000, size=batch_size))
    gsp_ids = np.random.random(size=batch_size)

    results = make_validation_results(
        predictions_mw=predictions,
        truths_mw=truths,
        t0_datetimes_utc=t0_datetimes_utc,
        batch_idx=0,
        gsp_ids=gsp_ids,
        capacity_mwp=capacity
    )

    assert len(results) == batch_size * forecast_length
    assert "t0_datetime_utc" in results.keys()
    assert "gsp_id" in results.keys()
    assert "actual_gsp_pv_outturn_mw" in results.keys()
    assert "forecast_gsp_pv_outturn_mw" in results.keys()


def test_save_validation_results_to_logger():
    """Test save_validation_results_to_logger"""
    batch_size = 2
    forecast_length = 4

    predictions = np.random.random(size=(batch_size, forecast_length))
    capacity = np.random.random(size=(batch_size, forecast_length))
    truths = np.random.random(size=(batch_size, forecast_length))
    t0_datetimes_utc = pd.to_datetime(np.random.randint(low=0, high=1000, size=batch_size))
    gsp_ids = np.random.random(size=batch_size)

    results1 = make_validation_results(
        predictions_mw=predictions,
        truths_mw=truths,
        t0_datetimes_utc=t0_datetimes_utc,
        batch_idx=0,
        gsp_ids=gsp_ids,
        capacity_mwp=capacity
    )

    results2 = make_validation_results(
        predictions_mw=predictions,
        truths_mw=truths,
        t0_datetimes_utc=t0_datetimes_utc,
        batch_idx=0,
        gsp_ids=gsp_ids,
        capacity_mwp=capacity
    )

    results_dfs = [results1, results2]

    save_validation_results_to_logger(
        results_dfs=results_dfs, logger=None, results_file_name="test_file_name", current_epoch=0
    )
