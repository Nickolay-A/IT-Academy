import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from .base import CustomTransformer


class RegexTransformer(CustomTransformer):

    def __init__(
            self,
            replace      : Optional[Dict[str, Optional[str]]] = None,
            find         : Optional[str] = None,
            split        : Optional[str] = None,
            target_dtype : Optional[type] = None,
            get_first    : bool = True,
        ):
        self.replace = replace
        self.find = find
        self.split = split
        self.target_dtype = target_dtype
        self.get_first = get_first

    def transform(self, X: pd.DataFrame, **params):
        X = super().transform(X)
        X = pd.concat(
                        [
                            self._process_series(column_values)
                            for _, column_values
                            in X.items()
                        ],
                        axis=1
                    )
        return X.values

    def _process_series(self, series: pd.Series) -> pd.Series:
        if self.replace is not None:
            for pat, repl in self.replace.items():
                if repl is None:
                    repl = lambda val: np.nan
                series = series.str.replace(pat, repl, case=False, regex=True)
        if self.find is not None:
            series = series.str.findall(self.find)
        if self.split is not None:
            series = series.str.split(self.split, regex=True)
        if self.get_first:
            series = series.str[0]
        if self.target_dtype is not None:
            series = series.astype(self.target_dtype)
        return series


class MultilabelEncoder(CustomTransformer):

    MISSING_INDICATOR = "<missing>"
    
    def __init__(self, split_by: Optional[str] = None):
        super().__init__()
        self.split_by = split_by
        self.encoders_ : Dict[str, MultiLabelBinarizer]
        self._output_as_dataframe: bool = False

    def set_output(self, *, transform=None):
        self._output_as_dataframe = transform == "pandas"
        return self

    def fit(self, X: pd.DataFrame, **params):
        super().fit(X, **params)
        # Создаём пустой словарь, где будем хранить трансформеры для каждого признака:
        self.encoders_ = {}
        # Перебираем каждый признак, создаём и обучаем транформер для него:
        for feature_name, feature_values in X.items():
            list_of_values = self._convert_to_lists(feature_values)
            feature_binarizer = MultiLabelBinarizer()
            feature_binarizer.fit(list_of_values)
            self.encoders_[feature_name] = feature_binarizer
        return self

    def transform(self, X: pd.DataFrame, **params):
        X = super().transform(X, **params)
        # Создаём лист где будем хранить кодированные значения:
        encodings = []
        # Фильтруем предупреждения о незнакомых значениях:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=UserWarning)
            for feature_name, binarizer in self.encoders_.items():
                feature_values = X[feature_name]
                list_of_values = self._convert_to_lists(feature_values)
                encodings.append(binarizer.transform(list_of_values))
        # Склеиваем все кодированные значения в один массив:
        result = np.concatenate(encodings, axis=1)
        if self._output_as_dataframe:
            return pd.DataFrame(result, columns=self.get_feature_names_out(), index=X.index)
        else:
            return result

    def _convert_to_lists(self, series: pd.Series) -> List[List[Any]]:
        if self.split_by is not None:
            series = series.str.split(self.split_by, regex=True)
        return [
                values
                if isinstance(values, list)
                else [self.MISSING_INDICATOR]
                for values
                in  series.values
            ]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(chain.from_iterable(
            [
                f"{feature_name}__{code}"
                for code
                in binarizer.classes_
            ]
            for feature_name, binarizer
            in  self.encoders_.items()
        ))
