import logging.config
import pickle

import click
import pandas as pd

from src.data import read_data

logger = logging.getLogger(__name__)


def predict(model: dict, data: pd.DataFrame) -> pd.Series:
    transformer = model["transformer"]
    classifier = model["classifier"]
    features = transformer.transform(data)
    return pd.Series(
        classifier.predict(features), index=data.index, name="prediction"
    )


@click.command()
@click.option("--data_path", default="data/raw/archive.zip")
@click.option("--model_path", default="models/model.pickle")
@click.option("--prediction_path", default="data/predictions/predictions.csv")
def main(data_path, model_path, prediction_path):
    logger.info(f"Data path: {data_path}, Model path: {model_path}, prediction_path: {prediction_path}")

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    data = read_data(data_path)

    if "target" in data:
        data = data.drop("target", axis=1)
    prediction = predict(model, data)

    logger.info(f"Prediction shape: {prediction.shape} ")

    prediction.to_csv(prediction_path)


if __name__ == "__main__":
    main()
