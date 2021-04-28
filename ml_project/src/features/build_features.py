import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.configs import FeatureParams, TransformerConfig
from src.features.transformer import FeatureScaleTransformer

logger = logging.getLogger(__name__)


def separate_target(
        data: pd.DataFrame, target_name: str
) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Started separating target")
    target = data[target_name]
    data_without_target = data.drop(target_name, axis=1)
    logger.info("Finished separating target")
    return data_without_target, target


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
    # return pd.DataFrame(transformer.transform(df).toarray())
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


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    if params.use_log_trick:
        target = pd.Series(np.log(target.to_numpy()))
    return target
