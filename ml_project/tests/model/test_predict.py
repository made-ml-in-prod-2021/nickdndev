import pandas as pd

from src.model.predict import predict


def test_predict(model: dict, dataset: pd.DataFrame, target_name: str):
    dataset = dataset.drop(target_name, axis=1)
    predictions = predict(model, dataset)
    assert len(predictions) == len(dataset)
    assert len(predictions.unique()) <= 2
