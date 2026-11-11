import numpy as np
import pandas as pd

from .registry import create_coder
from .schema import ColumnSpec


class TabCodec:
    def __init__(self, spec, output: str = 'array'):
        if output not in ('array', 'frame'):
            raise ValueError('output must be array or frame')
        self.spec = list(spec)
        self.output = output
        self.coders_ = {}

    def fit(self, df: pd.DataFrame, y=None) -> 'TabCodec':
        for s in self.spec:
            if s.name not in df.columns:
                raise KeyError(f'column {s.name!r} not found in dataframe')
            coder = create_coder(s.coder, **s.params)
            coder.fit(df[s.name], y)
            self.coders_[s.name] = coder
        return self

    @property
    def output_dim(self) -> int:
        return sum(c.output_dim for c in self.coders_.values)

    def _column_names(self):
        names = []
        for s in self.spec:
            d = self.coders_[s.name].output_dim
            if d == 1:
                names.append(s.name)
            else:
                names.extend(f'{s.name}_{i}' for i in range(d))
        return names