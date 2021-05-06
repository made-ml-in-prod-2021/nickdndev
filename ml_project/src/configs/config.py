from dataclasses import dataclass

from omegaconf import MISSING

from .app_config import AppConfig
from .data_profiling import DataProfilingConfig
from .model_config import ModelConfig
from .split_config import SplitConfig
from .transformer_config import TransformerConfig


@dataclass
class Config:
    split: SplitConfig = MISSING
    transformer: TransformerConfig = MISSING
    app: AppConfig = MISSING
    data_profiling: DataProfilingConfig = MISSING
    model: ModelConfig = MISSING
