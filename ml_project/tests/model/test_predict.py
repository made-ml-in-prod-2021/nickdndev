import pandas as pd

from src.model.predict_pipeline import predict

from src.model.trained_model import TrainedModel


def test_predict(model: TrainedModel, dataset: pd.DataFrame, target_col: str):
    dataset = dataset.drop(target_col, axis=1)
    predictions = predict(model, dataset)
    assert len(predictions) == len(dataset)
    assert len(predictions.unique()) <= 2
