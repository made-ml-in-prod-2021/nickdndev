import pandas as pd
import pytest
from pandas._testing import assert_series_equal
from sklearn.exceptions import NotFittedError

from src.configs import FeatureScale
from src.configs import TransformerConfig
from src.features import FeatureScaleTransformer
from src.features import split_target


def test_scale_feature_transformer(
        dataset: pd.DataFrame, transformer_config: TransformerConfig, target_col: str
):
    feature_scale_name = 'thalach'
    feature_scale = 2.

    transformer_config.feature_scale = FeatureScale(scale={feature_scale_name: feature_scale})
    data, target = split_target(dataset, target_col)
    transformer = FeatureScaleTransformer(transformer_config.feature_scale).fit(data)
    transformed_data = transformer.transform(data)
    assert_series_equal(transformed_data[feature_scale_name], feature_scale * data[feature_scale_name])


def test_scale_feature_transformer_invalid_config(
        dataset: pd.DataFrame, transformer_config: TransformerConfig, target_col: str
):
    with pytest.raises(Exception) as excinfo:
        feature_scale_name = 'thalach'
        feature_scale = -2.

        transformer_config.feature_scale = FeatureScale(scale={feature_scale_name: feature_scale})
        data, target = split_target(dataset, target_col)
        FeatureScaleTransformer(transformer_config.feature_scale).fit(data)
    assert str(excinfo.value) == 'Feature thalach scale <=0'


def test_scale_feature_transformer_not_fitted(
        dataset: pd.DataFrame, transformer_config: TransformerConfig, target_col: str
):
    with pytest.raises(NotFittedError):
        feature_scale_name = 'thalach'
        feature_scale = -2.

        transformer_config.feature_scale = FeatureScale(scale={feature_scale_name: feature_scale})
        data, target = split_target(dataset, target_col)
        transformer = FeatureScaleTransformer(transformer_config.feature_scale)
        transformer.transform(data)
