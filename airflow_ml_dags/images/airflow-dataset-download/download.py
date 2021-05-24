import pathlib

import click
from sklearn import datasets


# https://docs.seldon.io/projects/seldon-core/en/v1.1.0/examples/iris.html


def download_dataset(output_dir: str):
    x, y = datasets.load_iris(return_X_y=True, as_frame=True)

    dataset_dir = pathlib.Path(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    x.to_csv(dataset_dir / "data.csv", index=False)
    y.to_csv(dataset_dir / "target.csv", index=False)


@click.command("predict")
@click.option("--output-dir")
def download(output_dir: str):
    download_dataset(output_dir)


if __name__ == "__main__":
    download()
