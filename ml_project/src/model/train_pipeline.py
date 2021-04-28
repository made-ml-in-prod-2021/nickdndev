import logging.config
import os
import pickle
import typing
from pathlib import Path
from typing import Any

import hydra
import yaml
from sklearn.metrics import classification_report

from src.configs import Config, SplitConfig, TrainedModelConfig
from src.data import read_data, split_train_val_data
from src.features import separate_target
from src.features import build_transformer, make_features
from utils import get_path_from_root

logger = logging.getLogger(__name__)


def serialize_model(model: Any, cfg: TrainedModelConfig):
    path = get_path_from_root(cfg.model_dir)
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(path, cfg.model_name), "wb") as weights_file:
        pickle.dump(model, weights_file)


def save_metrics(metrics: dict, cfg: TrainedModelConfig):
    path = get_path_from_root(os.path.join(cfg.metric_dir))

    Path(path).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(path, cfg.metrics_file_name), "w") as metrics_file:
        metrics_file.write(yaml.dump(metrics))


def train_pipeline(cfg: Config) -> None:
    logger.info("Started train pipeline")
    data = read_data(get_path_from_root(cfg.app.input_data_path))

    logger.debug(f"Size dataset: {data.shape}")

    train_data, val_data = split_train_val_data(data, typing.cast(SplitConfig, cfg.split))

    train_features, train_target = separate_target(train_data, cfg.app.target_name)
    val_features, val_target = separate_target(val_data, cfg.app.target_name)

    logger.info(f"Transforming dataset.....")

    transformer = build_transformer(cfg.transformer)
    transformer.fit(train_features)

    train_features = make_features(transformer, train_features)
    logger.info(f"Train features shape :  {train_features.shape}")

    val_features = make_features(transformer, val_features)
    logger.info(f"Validation features shape :  {val_features.shape}")

    logger.info("Training model...")
    cls = hydra.utils.instantiate(cfg.model).fit(train_features, train_target)
    logger.info("Model is trained")

    logger.info("evaluating model...")

    val_predictions = cls.predict(val_features)
    metrics = classification_report(val_target, val_predictions, output_dict=True)
    logger.debug(f"Metrics: \n{yaml.dump(metrics)}")

    model = {"classifier": cls, "transformer": transformer}

    if cfg.app.trained_model.replace_model:
        logger.info("Start saving model")
        serialize_model(model, cfg.app.trained_model)
        logger.info(f"Finished saving model to {cfg.app.trained_model.model_dir}")

        logger.info("Start saving metrics")
        save_metrics(metrics, cfg.app.trained_model)
        logger.info(f"Finished saving metrics to {cfg.app.trained_model.metric_dir}")

    logger.info("Finished train pipeline")
