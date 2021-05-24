from pathlib import Path

import click
import pandas as pd


def preprocess_dataset(input_data_dir, output_data_dir):
    input_dir = Path(input_data_dir)

    x_df = pd.read_csv(input_dir / "data.csv")
    y_df = pd.read_csv(input_dir / "target.csv")

    output_dir = Path(output_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data_df = pd.concat([x_df, y_df], axis=1)
    train_data_df.to_csv(output_dir / "train_data", index=False)


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    preprocess_dataset(input_dir, output_dir)


if __name__ == "__main__":
    preprocess()
