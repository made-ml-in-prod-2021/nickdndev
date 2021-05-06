from dataclasses import dataclass

from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer


@dataclass
class TrainedModel:
    transformer: ColumnTransformer
    classifier: ClassifierMixin
