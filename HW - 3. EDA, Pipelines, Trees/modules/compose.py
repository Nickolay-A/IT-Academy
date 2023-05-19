import numpy as np
import pandas as pd
from typing import Union, Iterable, Dict, TypeAlias
from copy import deepcopy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .base import CustomTransformer


GroupTransformer: TypeAlias = Union[CustomTransformer, ColumnTransformer, Pipeline]


class GroupByTransformer(CustomTransformer):
    
    def __init__(
            self,
            groupby: Union[str, Iterable],
            transformer: GroupTransformer,
            stateless: bool = False,
        ):
        super().__init__()
        self.groupby = groupby
        self.transformer = transformer
        self.stateless = stateless
        self._output_as_dataframe: bool = False
        self.transformers_: Dict[str, GroupTransformer]

    def set_output(self, *, transform=None):
        self._output_as_dataframe = transform == "pandas"
        self.transformer.set_output(transform=transform)
        return self

    def fit(self, X: pd.DataFrame, **params):
        super().fit(X, **params)
        if not self.stateless:
            self.transformers_ = {
                                    group_name: deepcopy(self.transformer).fit(group_values, **params)
                                    for group_name, group_values
                                    in X.groupby(self.groupby)
                                }
        return self
    
    def transform(self, X: pd.DataFrame, **params):
        X = super().transform(X, copy=True, **params)
        # Temporary add row_id to X:
        X.index = pd.MultiIndex.from_frame(X.index.to_frame().assign(temp__row_id=np.arange(len(X))))
        X_transformed = pd.concat([
                                    self._transform_group(group_name, group_values, **params)
                                    for group_name, group_values
                                    in X.groupby(self.groupby, group_keys=True, as_index=False)
                                ])
        # Return to original order:
        X_transformed = X_transformed.sort_index("temp__row_id")
        # Drop temporary row_id:
        X_transformed.index = X_transformed.index.droplevel("temp__row_id")
        if self._output_as_dataframe:
            return X_transformed
        else:
            return X_transformed.values

    def _transform_group(self, group_name: str, group_values: pd.DataFrame, **params) -> pd.DataFrame:
        if self.stateless:
            values = deepcopy(self.transformer).fit_transform(group_values, **params)
        else:
            values = self.transformers_[group_name].transform(group_values, **params)
        if isinstance(values, np.ndarray):
            values = pd.DataFrame(values, columns=group_values.columns)
        values.index = group_values.index
        return values
