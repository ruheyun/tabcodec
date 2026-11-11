import numpy as np
import pandas as pd

from ..base import BaseCoder


class StandardScalerCoder(BaseCoder):
    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0

    def fit(self, X, y=None) -> 'StandardScalerCoder':
        s = pd.to_numeric(pd.Series(X), errors='coerce')
        self.mean_ = float(s.mean())
        std = float(s.std(ddof=0))
        self.std_ = std if np.isfinite(std) and std != 0 else 1.0
        return self

    @property
    def output_dim(self) -> int:
        return 1

    def transform(self, X) -> np.ndarray:
        s = pd.to_numeric(pd.Series(X), errors='coerce').to_numpy(dtype=float)
        return ((s - self.mean_) / self.std_).reshape(-1, 1)

    def inverse_transform(self, Z) -> pd.Series:
        z = np.asarray(Z, dtype=float).reshape(-1)
        return pd.Series(z * self.std_ + self.mean_)