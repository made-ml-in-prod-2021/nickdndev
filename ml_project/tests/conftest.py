import os

import pandas as pd
import pytest

from data_generator import generate_dataset
from ml_project.configs import SplitConfig
from ml_project.data import read_data


@pytest.fixture()
def dataset_path() -> str:
    path = os.path.join(os.path.dirname(__file__), "data_sample.zip")
    data = generate_dataset()
    data.to_csv(path, compression="zip")
    return path


@pytest.fixture()
def dataset(dataset_path) -> pd.DataFrame:
    return read_data(dataset_path)


@pytest.fixture()
def target_col() -> str:
    return "restecg"


@pytest.fixture()
def split_config() -> SplitConfig:
    return SplitConfig(val_size=0.2, random_state=42)
