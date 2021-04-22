from dataclasses import dataclass
from .split_config import SplitConfig
from omegaconf import MISSING


@dataclass
class Config:
    split: SplitConfig = MISSING
