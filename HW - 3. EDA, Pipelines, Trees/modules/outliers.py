import numpy as np
from typing import Optional
from .base import CustomTransformer


class OutlierDetector:

    def __init__(self, clip: bool = False):
        self.clip = clip
        self.data_min: np.ndarray
        self.data_max: np.ndarray

    def transform(self, X: np.ndarray, **params):
        if self.clip:
            return self._clip_outliers(X)
        else:
            return self._remove_outliers(X)

    def _clip_outliers(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.data_min, self.data_max)

    def _remove_outliers(self, X: np.ndarray) -> np.ndarray:
        return np.where((X >= self.data_min) & (X <= self.data_max), X, np.nan)


class RangeOutlierDetector(CustomTransformer, OutlierDetector):

    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None, clip: bool = False):
        CustomTransformer.__init__(self)
        OutlierDetector.__init__(self, clip=clip)
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None, **params):
        super().fit(X)
        self.data_min = np.asarray(X.min(axis=0)) if self.min_value is None else np.full(X.shape[1], self.min_value)
        self.data_max = np.asarray(X.max(axis=0)) if self.max_value is None else np.full(X.shape[1], self.max_value)
        return self

    def transform(self, X, **params):
        X = CustomTransformer.transform(self, X)
        X = OutlierDetector.transform(self, X)
        return X


class IQROutlierDetector(CustomTransformer, OutlierDetector):

    def __init__(self, quantile_min: float = .25, quantile_max: float = .75, iqr_multiplier: float = 1.5, clip: bool = False):
        CustomTransformer.__init__(self)
        OutlierDetector.__init__(self, clip=clip)
        self.quantile_min = quantile_min
        self.quantile_max = quantile_max
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None, **params):
        super().fit(X)
        q_min = np.nanquantile(X, self.quantile_min, axis=0)
        q_max = np.nanquantile(X, self.quantile_max, axis=0)
        iqr = q_max - q_min
        self.data_min = q_min - self.iqr_multiplier * iqr
        self.data_max = q_max + self.iqr_multiplier * iqr
        return self

    def transform(self, X, **params):
        X = CustomTransformer.transform(self, X)
        X = OutlierDetector.transform(self, X)
        return X


class DeltaFromMedianOutlierDetector(CustomTransformer, OutlierDetector):

    def __init__(self, delta: float = 3, clip: bool = False):
        CustomTransformer.__init__(self)
        OutlierDetector.__init__(self, clip=clip)
        self.delta = delta

    def fit(self, X, y=None, **params):
        super().fit(X)
        median = np.nanmedian(X, axis=0)
        self.data_min = median / self.delta
        self.data_max = median * self.delta
        return self

    def transform(self, X, **params):
        X = CustomTransformer.transform(self, X)
        X = OutlierDetector.transform(self, X)
        return X
