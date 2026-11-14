import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from ..base import BaseCoder


class SklearnNumericCoder(BaseCoder):
    def __init__(self, method: str = "standard", allow_nan: bool = False, **kwargs):
        self.method = method
        self.allow_nan = allow_nan

        if method == "standard":
            self.scaler = StandardScaler(**kwargs)
        elif method == "minmax":
            self.scaler = MinMaxScaler(**kwargs)
        elif method == "robust":
            self.scaler = RobustScaler(**kwargs)
        elif method == "quantile":
            self.scaler = QuantileTransformer(**kwargs)
        elif method == "power":
            self.scaler = PowerTransformer(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

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
        

    def fit(self, X, y=None) -> 'SklearnNumericCoder':
        raw = pd.to_numeric(pd.Series(X), errors='coerce').to_numpy(dtype=float).reshape(-1, 1)
        self._check_nan(raw)
        self.scaler.fit(raw)
        return self

    @property
    def output_dim(self) -> int:
        return 1

    def transform(self, X) -> np.ndarray:
        raw = pd.to_numeric(pd.Series(X), errors='coerce').to_numpy(dtype=float).reshape(-1, 1)
        self._check_nan(raw)
        return self.scaler.transform(raw)

    def inverse_transform(self, Z) -> pd.Series:
        Z = np.asarray(Z, dtype=float)
        return pd.Series(self.scaler.inverse_transform(Z.reshape(-1, 1)).reshape(-1))