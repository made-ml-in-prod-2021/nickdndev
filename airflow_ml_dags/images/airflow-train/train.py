import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(input_data_dir: str, output_model_dir: str):
    input_data_path = Path(input_data_dir)
    train_df = pd.read_csv(input_data_path / "train_data.csv")

    model = LogisticRegression()
    model.fit()

    output_model_path = Path(output_model_dir)
    output_model_path.mkdir(exist_ok=True, parents=True)

    output_model_path = str(output_model_dir / "model")
    with open(output_model_path, "wb") as weights_file:
        pickle.dump(model, weights_file)


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    train_model(input_dir, output_dir)


if __name__ == "__main__":
    train()
