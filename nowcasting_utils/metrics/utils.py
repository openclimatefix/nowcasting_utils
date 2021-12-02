""" Util evaluation functions """
import pandas as pd


def check_results_df(results_df: pd.DataFrame):
    """
    Check the dataframe has the correct columns

    Args:
        results_df: results dataframe

    """

    assert len(results_df) > 0
    assert "t0_datetime_utc" in results_df.keys()
    assert "target_datetime_utc" in results_df.keys()
    assert "forecast_gsp_pv_outturn_mw" in results_df.keys()
    assert "actual_gsp_pv_outturn_mw" in results_df.keys()
    assert "gsp_id" in results_df.keys()
    assert "capacity_mwp" in results_df.keys()
