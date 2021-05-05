import os

import pytest


@pytest.fixture()
def dataset_path() -> str:
    return 'data/heart.csv'


@pytest.fixture()
def model_path() -> str:
    os.environ['PATH_TO_MODEL'] = 'model/model.pkl'
    return os.getenv("PATH_TO_MODEL")
