import logging.config

import pandas as pd

from src.model.trained_model import TrainedModel

logger = logging.getLogger(__name__)


def predict(model: TrainedModel, data: pd.DataFrame) -> pd.Series:
    logger.info(f"Data shape: {data.shape}")

    features = model.transformer.transform(data)

    predictions = pd.Series(
        model.classifier.predict(features), index=data.index, name="prediction"
    )
    logger.info(f"Prediction shape: {predictions.shape}")

    return predictions
