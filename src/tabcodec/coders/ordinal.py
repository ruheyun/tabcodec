import numpy as np
import pandas as pd

from ..base import BaseCoder


_NAN = object()


class OrdinalCoder(BaseCoder):
    def __init__(self):
        self.categories_ = []
        self._cat_to_idx = {}

    def fit(self, X, y=None) -> "OrdinalCoder":
        cats = []
        for v in pd.unique(pd.Series(X)):
            cats.append(_NAN if pd.isna(v) else v)
        self.categories_ = cats
        self._cat_to_idx = {cat: i for i, cat in enumerate(cats)}
        return self

    @property
    def output_dim(self) -> int:
        return 1

    def transform(self, X) -> np.ndarray:
        s = pd.Series(X).map(lambda v: _NAN if pd.isna(v) else v)
        return s.map(self._cat_to_idx).to_numpy().reshape(-1, 1)

    def inverse_transform(self, Z) -> pd.Series:
        idx = np.asarray(Z).reshape(-1)
        values = []
        for v in idx:
            if v < 0 or v >= len(self.categories_):
                values.append(np.nan)
            else:
                c = self.categories_[int(v)]
                values.append(np.nan if c is _NAN else c)
        return pd.Series(values)