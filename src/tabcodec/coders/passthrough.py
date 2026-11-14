import numpy as np
import pandas as pd

from ..base import BaseCoder


class PassthroughCoder(BaseCoder):
    def fit(self, X, y=None) -> "PassthroughCoder":
        return self

    @property
    def output_dim(self) -> int:
        return 1

    def transform(self, X) -> np.ndarray:
        return np.asarray(X).reshape(-1, 1)

    def inverse_transform(self, Z) -> pd.Series:
        return pd.Series(np.asarray(Z).reshape(-1))