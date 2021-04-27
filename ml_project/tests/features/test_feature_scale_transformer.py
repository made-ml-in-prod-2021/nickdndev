import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from ml_project.configs import TransformerConfig
from ml_project.configs.transformer_config import FeatureScale
from ml_project.features import FeatureScaleTransformer
from ml_project.features import separate_target


def test_scale_feature_transformer(
        dataset: pd.DataFrame, transformer_config: TransformerConfig, target_name: str
):
    feature_scale_name = 'thalach'
    feature_scale = 2.

    transformer_config.feature_scale = FeatureScale(scale={feature_scale_name: feature_scale})
    data, target = separate_target(dataset, target_name)
    transformer = FeatureScaleTransformer(transformer_config.feature_scale).fit(data)
    transformed_data = transformer.transform(data)
    assert_series_equal(transformed_data[feature_scale_name], feature_scale * data[feature_scale_name])


def test_scale_feature_transformer_invalid_config(
        dataset: pd.DataFrame, transformer_config: TransformerConfig, target_name: str
):
    with pytest.raises(Exception) as excinfo:
        feature_scale_name = 'thalach'
        feature_scale = -2.

        transformer_config.feature_scale = FeatureScale(scale={feature_scale_name: feature_scale})
        data, target = separate_target(dataset, target_name)
        FeatureScaleTransformer(transformer_config.feature_scale).fit(data)
    assert str(excinfo.value) == 'Feature thalach scale <=0'
