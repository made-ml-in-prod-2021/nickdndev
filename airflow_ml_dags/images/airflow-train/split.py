from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_data(input_dir: str):
    input_data_path = Path(input_dir)
    train_data_df = pd.read_csv(input_data_path / "train_data.csv")
    train_df, val_df = train_test_split(train_data_df, test_size=0.2)

    train_df.to_csv(input_data_path / "train_data.csv", index=False)
    val_df.to_csv(input_data_path / "dev_data.csv", index=False)


@click.command("preprocess")
@click.option("--input-dir")
def split(input_dir: str):
    split_train_data(input_dir)


if __name__ == "__main__":
    split()
