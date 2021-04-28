import os

import yaml

from src.configs import Config
from src.model.train_pipeline import train_pipeline


def test_train_pipeline(train_config: Config):
    expected_model_path = os.path.join(train_config.app.trained_model.model_dir,
                                       train_config.app.trained_model.model_name)

    expected_metrics_path = os.path.join(train_config.app.trained_model.metric_dir,
                                         train_config.app.trained_model.metrics_file_name)

    train_pipeline(train_config)

    assert os.path.exists(expected_model_path)
    assert os.path.exists(expected_metrics_path)

    with open(expected_metrics_path, "r") as metrics_file:
        metrics = yaml.load(metrics_file, Loader=yaml.FullLoader)
        assert metrics["accuracy"] > 0
