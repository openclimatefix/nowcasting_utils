from nowcasting_utils.metrics.evaluation import evaluation
import nowcasting_utils

import pandas as pd
from pathlib import Path


def test_evaluation():

    file = Path(nowcasting_utils.__file__).parent.parent.absolute() / "tests" / "metrics" / "epoch_0.csv"

    data_df = pd.read_csv(file, index_col=[0])
    data_df['t0_datetime_utc'] = pd.to_datetime(data_df['t0_datetime_utc'])
    data_df['target_datetime_utc'] = pd.to_datetime(data_df['target_datetime_utc'])

    evaluation(data_df, 'Unittest Model Name')