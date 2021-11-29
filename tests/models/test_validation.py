""" Test saving validation results """
import numpy as np

from nowcasting_utils.models.validation import (
    make_validation_results,
    save_validation_results_to_logger,
)


def test_make_validation_results():
    """ Test making the validation """

    batch_size = 2
    forecast_length = 4

    predictions = np.random.random(size=(batch_size, forecast_length))
    truths = np.random.random(size=(batch_size, forecast_length))
    t0_datetimes_utc = np.random.random(size=batch_size)
    gsp_ids = np.random.random(size=batch_size)

    results = make_validation_results(
        predictions=predictions,
        truths=truths,
        t0_datetimes_utc=t0_datetimes_utc,
        batch_idx=0,
        gsp_ids=gsp_ids,
    )

    assert len(results) == batch_size
    assert 't0_datetime_utc' in results.keys()
    assert 'gsp_id' in results.keys()
    for i in range(forecast_length):
        assert f'truth_{i}' in results.keys()
        assert f'prediction_{i}' in results.keys()


def test_save_validation_results_to_logger():
    """ Test save_validation_results_to_logger """
    batch_size = 2
    forecast_length = 4

    predictions = np.random.random(size=(batch_size, forecast_length))
    truths = np.random.random(size=(batch_size, forecast_length))
    t0_datetimes_utc = np.random.random(size=batch_size)
    gsp_ids = np.random.random(size=batch_size)

    results1 = make_validation_results(
        predictions=predictions,
        truths=truths,
        t0_datetimes_utc=t0_datetimes_utc,
        batch_idx=0,
        gsp_ids=gsp_ids,
    )

    results2 = make_validation_results(
        predictions=predictions,
        truths=truths,
        t0_datetimes_utc=t0_datetimes_utc,
        batch_idx=0,
        gsp_ids=gsp_ids,
    )

    results_dfs = [results1, results2]

    save_validation_results_to_logger(results_dfs=results_dfs, logger=None,results_file_name = 'test_file_name', current_epoch=0)

