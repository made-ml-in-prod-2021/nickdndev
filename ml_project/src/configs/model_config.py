from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    _target_: str = MISSING


@dataclass
class LogisticRegressionConfig(ModelConfig):
    random_state: int = MISSING
    C: float = MISSING
    max_iter: int = MISSING
