import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.configs import SplitConfig

log = logging.getLogger(__name__)


def read_data(path: str) -> pd.DataFrame:
    log.info(f"Started reading data from {path}")
    data = pd.read_csv(path)
    log.info(f"Finished reading data")
    return data


def split_train_val_data(
        data: pd.DataFrame, split_config: SplitConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    log.info("Started splitting data to train and val ")
    train_data, val_data = train_test_split(
        data, test_size=split_config.val_size, random_state=split_config.random_state
    )
    log.info("Finished splitting data to train and val")
    return train_data, val_data
