import numpy as np
import pandas as pd

from ..base import BaseCoder


_NAN = object()


class OneHotCoder(BaseCoder):
    def __init__(self, drop: str | None = None):
        if drop not in (None, 'first'):
            raise ValueError('drop must be None or first')
        self.drop = drop
        self.categories_: list = []
        self._cat_to_idx: dict = {}

    def fit(self, X, y=None) -> 'OneHotCoder':
        s = pd.Series(X)
        cats = []
        for v in pd.unique(s):
            if pd.isna(v):
                cats.append(_NAN)
            else:
                cats.append(v)
        self.categories_ = cats
        self._cat_to_idx = {c: i for i, c in enumerate(cats)}
        return self

    @property
    def n_categories(self) -> int:
        return len(self.categories_)

    @property
    def output_dim(self) -> int:
        return self.n_categories - (1 if self.drop == 'first' else 0)

    def transform(self, X) -> np.ndarray:
        s = pd.Series(X)
        out = np.zeros((len(s), self.n_categories), dtype=np.float32)
        for i, v in enumerate(s):
            key = _NAN if pd.isna(v) else v
            idx = self._cat_to_idx.get(key, -1)
            if idx != -1:
                out[i, idx] = 1.0
        return out[:, 1:] if self.drop == 'first' else out

    def inverse_transform(self, Z) -> pd.Series:
        Z = np.asarray(Z)
        if self.drop == 'first':
            pad = np.zeros((Z.shape[0], 1), dtype=Z.dtype)
            pad[Z.sum(axis=1) == 0, 0] = 1.0
            Z_full = np.hstack([pad, Z])
        else:
            Z_full = Z

        idx = Z_full.argmax(axis=1)
        unknown = Z_full.sum(axis=1) == 0
        values = []
        for i, j in enumerate(idx):
            if unknown[i]:
                values.append(np.nan)
            else:
                c = self.categories_[j]
                values.append(np.nan if c is _NAN else c)

        return pd.Series(values)
    
