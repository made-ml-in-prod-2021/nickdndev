from hydra.core.config_store import ConfigStore

from .config import Config
from .app_config import TrainedModelConfig, AppConfig
from .model_config import LogisticRegressionConfig
from .split_config import SplitConfig

__all__ = ["Config", "TransformerConfig", "SplitConfig", "TrainedModelConfig"]

from .transformer_config import TransformerConfig

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="app", name="app", node=AppConfig)
cs.store(group="split", name="split", node=SplitConfig)

cs.store(group="model", name="logistic_regression", node=LogisticRegressionConfig)
