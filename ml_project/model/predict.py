import pickle

import click
import pandas as pd

from ml_project.data import read_data


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
@click.option("--output_path", default="data/preds/predictions.csv")
def main(data_path, model_path, output_path):
    data = read_data(data_path)
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    if "target" in data:
        data = data.drop("target", axis=1)
    predictions = predict(model, data)
    predictions.to_csv(output_path)


if __name__ == "__main__":
    main()
