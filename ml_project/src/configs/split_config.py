from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class SplitConfig:
    name: str = MISSING
    val_size: float = MISSING
    random_state: int = MISSING
