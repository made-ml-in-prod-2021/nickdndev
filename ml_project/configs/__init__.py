from hydra.core.config_store import ConfigStore
from .config import Config
from .split_config import SplitConfig

__all__ = ["Config", "SplitConfig"]

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="split", name="split", node=SplitConfig)
