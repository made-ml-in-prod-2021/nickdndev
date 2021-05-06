import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.configs import TransformerConfig
from src.features.transformer import FeatureScaleTransformer

logger = logging.getLogger(__name__)


def split_target(
        data: pd.DataFrame, target_name: str
) -> Tuple[pd.DataFrame, pd.Series]:
    return data.drop(target_name, axis=1), data[target_name]


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    return pd.DataFrame(categorical_pipeline.fit_transform(categorical_df).toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ("ohe", OneHotEncoder()),
    ])
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    num_pipeline = build_numerical_pipeline()
    return pd.DataFrame(num_pipeline.fit_transform(numerical_df))


def build_numerical_pipeline(transfer_config: TransformerConfig) -> Pipeline:
    numerical_pipeline = Pipeline([
        ("feature scale", FeatureScaleTransformer(transfer_config.feature_scale)),
        ("scaler", StandardScaler())
    ])
    return numerical_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df))


def build_transformer(transfer_config: TransformerConfig) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                [c for c in transfer_config.feature_params.categorical_features],
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(transfer_config),
                [n for n in transfer_config.feature_params.numerical_features],
            ),
        ]
    )
    return transformer
