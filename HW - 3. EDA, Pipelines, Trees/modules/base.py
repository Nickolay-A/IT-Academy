from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        self.feature_names_in_: List[str]

    def fit(self, X, y=None, **params) -> 'CustomTransformer':
        self.feature_names_in_ = list(X.columns)
        self.n_features_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, X, copy: bool = False, **params):
        check_is_fitted(self, 'is_fitted_')
        if copy:
            X = X.copy()
        return X

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return self.feature_names_in_
