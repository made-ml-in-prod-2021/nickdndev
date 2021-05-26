from pathlib import Path

import click
import pandas as pd


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    input_data_path = Path(input_dir)

    x_df = pd.read_csv(input_data_path / "data.csv")
    y_df = pd.read_csv(input_data_path / "target.csv")

    y_df.set_axis(["target"], axis=1)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    data_processed = pd.concat([x_df, y_df], axis=1)
    data_processed.to_csv(output_dir_path / "train_data.csv", index=False)


if __name__ == '__main__':
    preprocess()
