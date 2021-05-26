from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
def split(input_dir: str):
    input_data_path = Path(input_dir)
    train_data_df = pd.read_csv(input_data_path / "train_data.csv")
    train_df, val_df = train_test_split(train_data_df, test_size=0.2)

    train_df.to_csv(input_data_path / "train.csv", index=False)
    val_df.to_csv(input_data_path / "val.csv", index=False)


if __name__ == '__main__':
    split()
