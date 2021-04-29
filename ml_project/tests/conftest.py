import os
import pickle
from typing import cast, List, Dict

import pandas as pd
import pytest
from hydra.experimental import compose, initialize

from tests.data_generator import generate_dataset
from src.configs import SplitConfig, TransformerConfig, Config
from src.configs import FeatureParams, FeatureScale
from src.data import read_data
from src.model.train_pipeline import train_pipeline


@pytest.fixture()
def dataset_path() -> str:
    path = os.path.join(os.path.dirname(__file__),  "dataset.zip")
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


@pytest.fixture()
def target_name() -> str:
    return "target"


@pytest.fixture()
def model_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "model_test_dir")


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal',
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca',
    ]


@pytest.fixture()
def features_to_drop() -> List[str]:
    return []


@pytest.fixture()
def feature_scale() -> Dict[str, float]:
    return {}


@pytest.fixture()
def transformer_config(categorical_features, numerical_features, target_col,
                       features_to_drop, feature_scale) -> TransformerConfig:
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_col=target_col,
        features_to_drop=features_to_drop
    )

    feature_scale = FeatureScale(scale=feature_scale)
    transformer_config = TransformerConfig(
        feature_params=feature_params,
        feature_scale=feature_scale
    )
    return transformer_config


@pytest.fixture()
def train_config(model_dir, dataset_path, categorical_features, numerical_features, target_col,
                 features_to_drop) -> Config:
    try:
        initialize(config_path="../conf", job_name="test_app")
    except ValueError:
        pass
    cfg = compose(config_name="config")
    cfg = cast(Config, cfg)
    cfg.app.trained_model.metric_dir = model_dir
    cfg.app.trained_model.model_dir = model_dir
    cfg.app.trained_model.replace_model = True
    cfg.app.input_data_path = dataset_path

    return cfg


@pytest.fixture()
def model(train_config):
    train_pipeline(train_config)
    model_path = os.path.join(
        train_config.app.trained_model.model_dir,
        train_config.app.trained_model.model_name,
    )
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model
