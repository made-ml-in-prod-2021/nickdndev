from dataclasses import dataclass, field
from typing import List, Optional, Dict

from omegaconf import MISSING


@dataclass()
class FeatureParams:
    categorical_features: List[str] = MISSING
    numerical_features: List[str] = MISSING
    features_to_drop: List[str] = MISSING
    target_col: Optional[str] = MISSING
    use_log_trick: bool = field(default=True)


@dataclass()
class FeatureScale:
    scale: Dict[str, float] = MISSING


@dataclass
class TransformerConfig:
    feature_params: FeatureParams
    feature_scale: FeatureScale
