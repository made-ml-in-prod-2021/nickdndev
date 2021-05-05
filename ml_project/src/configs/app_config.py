from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class TrainedModelConfig:
    model_dir: str = MISSING
    model_name: str = MISSING
    replace_model: bool = MISSING
    metric_dir: str = MISSING
    metrics_file_name: str = MISSING


@dataclass
class AppConfig:
    input_data_dir: str = MISSING
    prediction_name: str = MISSING
    prediction_path: str = MISSING
    random_seed: int = MISSING
    trained_model: TrainedModelConfig = MISSING
