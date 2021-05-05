import logging.config
import os
import pickle

import hydra
from src.configs import Config
from src.data import read_data
from src.model.predict_pipeline import predict
from src.utils import construct_abs_path

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: Config):
    logger.info(
        f"Data path: {cfg.app.input_data_path}, Model path: {cfg.app.trained_model.model_dir}, prediction_path: {cfg.app.prediction_dir}")
    model_path = construct_abs_path(os.path.join(cfg.app.trained_model.model_dir, cfg.app.trained_model.model_name))
    data_path = construct_abs_path(cfg.app.input_data_path)
    prediction_path = construct_abs_path(os.path.join(cfg.app.prediction_dir, cfg.app.prediction_name))

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    data = read_data(data_path)

    if "target" in data:
        data = data.drop("target", axis=1)
    prediction = predict(model, data)

    logger.info(f"Prediction shape: {prediction.shape} ")

    prediction.to_csv(prediction_path)


if __name__ == "__main__":
    main()
