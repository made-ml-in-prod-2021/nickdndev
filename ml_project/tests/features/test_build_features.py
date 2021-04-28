import pandas as pd

from src.configs import TransformerConfig
from src.data import read_data
from src.features import separate_target
from src.features import build_transformer, make_features


def test_separate_target(dataset: pd.DataFrame, target_name: str):
    data_without_target, target = separate_target(dataset, target_name)
    assert len(data_without_target) > 0
    assert len(data_without_target) == len(target)


def test_make_features(
        transformer_config: TransformerConfig, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = build_transformer(transformer_config)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in transformer_config.feature_params.features_to_drop)
