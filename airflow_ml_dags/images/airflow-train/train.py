import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):

    input_data_path = Path(input_dir)
    dataset_df = pd.read_csv(input_data_path / "train.csv")

    y = dataset_df[['target']]
    x = dataset_df.drop(['target'], axis=1)

    model = LogisticRegression()
    model.fit(x, y)

    output_model_path = Path(output_dir)
    output_model_path.mkdir(exist_ok=True, parents=True)

    output_model_path = str(output_model_path / "model")
    with open(output_model_path, "wb") as weights_file:
        pickle.dump(model, weights_file)


if __name__ == "__main__":
    train()
