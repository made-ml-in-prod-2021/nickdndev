import logging.config

import pandas as pd

logger = logging.getLogger(__name__)


def predict(model: dict, data: pd.DataFrame) -> pd.Series:
    logger.info(f"Data shape: {data.shape}")
    transformer = model["transformer"]
    classifier = model["classifier"]
    features = transformer.transform(data)
    return pd.Series(
        classifier.predict(features), index=data.index, name="prediction"
    )