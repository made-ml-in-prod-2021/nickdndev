import pandas as pd

from ml_project.configs.transformer_config import TransformerConfig
from ml_project.data import read_data
from ml_project.features import separate_target
from ml_project.features.build_features import build_transformer, make_features


def test_separate_target(dataset: pd.DataFrame, target_name: str):
    data_without_target, target = separate_target(dataset, target_name)
    assert len(data_without_target) > 0
    assert len(data_without_target) == len(target)


#
# @pytest.fixture
# def feature_params(
#         categorical_features: List[str],
#         features_to_drop: List[str],
#         numerical_features: List[str],
#         target_col: str,
# ) -> FeatureParams:
#     params = FeatureParams(
#         categorical_features=categorical_features,
#         numerical_features=numerical_features,
#         features_to_drop=features_to_drop,
#         target_col=target_col,
#         use_log_trick=True,
#     )
#     return params


def test_make_features(
        transformer_config: TransformerConfig, dataset_path: str,
):
    data = read_data(dataset_path)
    transformer = build_transformer(transformer_config)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in transformer_config.feature_params.features_to_drop)
