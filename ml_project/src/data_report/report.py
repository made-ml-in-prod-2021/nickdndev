import logging.config
import os
from pathlib import Path

import hydra
from pandas_profiling import ProfileReport

from src.configs import Config
from src.data import read_data
from src.utils import construct_abs_path

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: Config) -> None:
    logger.info("Starting data profiling...")

    data = read_data(construct_abs_path(cfg.data_profiling.input_data_path))
    prof = ProfileReport(data)

    data_profiling_dir = construct_abs_path(cfg.data_profiling.report_dir)
    Path(data_profiling_dir).mkdir(parents=True, exist_ok=True)
    prof.to_file(os.path.join(data_profiling_dir, cfg.data_profiling.report_file_name))

    logger.info("Finished data profiling")


if __name__ == "__main__":
    main()
