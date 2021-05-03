import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.configs import FeatureScale


class FeatureScaleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fs: FeatureScale):
        self.fs = fs

    def fit(self, x: pd.DataFrame):
        for feature, scale in self.fs.scale.items():
            if scale <= 0:
                raise Exception(f"Feature {feature} scale <=0")
        return self

    def transform(self, x: pd.DataFrame):
        x_copy = x.copy()
        for feature, scale in self.fs.scale.items():
            x_copy[feature] = scale * x_copy[feature]
        return x_copy
