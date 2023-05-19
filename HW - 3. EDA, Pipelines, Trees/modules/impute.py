import pandas as pd
from .base import CustomTransformer


class InterpolationImputer(CustomTransformer):

    def __init__(self, method: 'str' = 'linear'):
        self.method = method

    def fit(self, X: pd.DataFrame, **params):
        super().fit(X, **params)
        return self

    def transform(self, X: pd.DataFrame, **params):
        X = super().transform(X, copy=True, **params)
        X = pd.concat(
                        [
                            feature_values.interpolate(method=self.method, limit_direction="both", downcast="infer")
                            for _, feature_values
                            in  X.items()
                        ],
                        axis=1
                    )
        return X.values
