import pandas as pd

from src.configs import SplitConfig
from src.data import read_data, split_train_val_data


def test_load_data(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_train_val_data(dataset: pd.DataFrame, split_config: SplitConfig):
    train_data, val_data = split_train_val_data(dataset, split_config)
    assert len(train_data) > 0
    assert len(val_data) > 0
