import numpy as np
import pandas as pd

from ..base import BaseCoder


class StandardScalerCoder(BaseCoder):
    def __init__(self, allow_nan: bool = False):
        self.allow_nan = allow_nan
        self.mean_ = 0.0
        self.std_ = 1.0

    def _check_nan(self, raw: np.ndarray):
        if self.allow_nan:
            return
        mask = np.isnan(raw)
        if mask.any():
            bad = np.flatnonzero(mask)[:5].tolist()
            raise ValueError(
                f"Input contains NaN values at indices {bad} (first 5 shown). "
                "Set allow_nan=True to allow NaN values."
            )
        

    def fit(self, X, y=None) -> 'StandardScalerCoder':
        raw = pd.to_numeric(pd.Series(X), errors='coerce').to_numpy(dtype=float)
        self._check_nan(raw)
        if self.allow_nan:
            self.mean_ = float(np.nanmean(raw)) if len(raw) else 0.0
            std = float(np.nanstd(raw, ddof=0)) if len(raw) else 1.0
        else:
            self.mean_ = float(np.mean(raw)) if len(raw) else 0.0
            std = float(np.std(raw, ddof=0)) if len(raw) else 1.0
        self.std_ = std if np.isfinite(std) and std > 0 else 1.0
        return self

    @property
    def output_dim(self) -> int:
        return 1

    def transform(self, X) -> np.ndarray:
        raw = pd.to_numeric(pd.Series(X), errors='coerce').to_numpy(dtype=float)
        self._check_nan(raw)
        return ((raw - self.mean_) / self.std_).reshape(-1, 1)

    def inverse_transform(self, Z) -> pd.Series:
        z = np.asarray(Z, dtype=float).reshape(-1)
        return pd.Series(z * self.std_ + self.mean_)