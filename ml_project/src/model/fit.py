import logging.config

import hydra

from src.configs import Config
from src.model.train_pipeline import train_pipeline

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: Config):
    train_pipeline(cfg)


if __name__ == "__main__":
    main()
