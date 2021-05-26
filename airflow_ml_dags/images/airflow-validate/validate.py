import json
import pickle
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import accuracy_score


@click.command("validate")
@click.option("--input-dir")
@click.option("--input-model-dir")
def validate(input_model_dir: str, input_dir: str):
    input_model_path = Path(input_model_dir)
    input_dataset_path = Path(input_dir)

    with open(input_model_path / "model", "rb") as f:
        model = pickle.load(f)

    val_df = pd.read_csv(input_dataset_path / "val.csv")
    y_val_df = val_df[['target']]
    x_val_df = val_df.drop(['target'], axis=1)

    y = model.predict(x_val_df)

    metrics = {
        "accuracy_score": accuracy_score(y_val_df.values, y),
    }

    with open(input_model_path / "metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    validate()
