import pickle
from pathlib import Path

import click
import pandas as pd


@click.command("predict")
@click.option("--input-dir")
@click.option("--input-model-dir")
@click.option("--output-dir")
def predict(input_dir: str, input_model_dir: str, output_dir: str):
    input_data_path = Path(input_dir)
    input_model_path = Path(input_model_dir)
    output_dir_path = Path(output_dir)

    with open(input_model_path / "model", "rb") as f:
        model = pickle.load(f)

    data_df = pd.read_csv(input_data_path / "data.csv")

    predicts = model.predict(data_df)

    predicts_df = pd.DataFrame(predicts, columns=["predictions"])

    output_dir_path.parent.mkdir(parents=True, exist_ok=True)
    predicts_df.to_csv(output_dir_path, index=False)


if __name__ == "__main__":
    predict()
