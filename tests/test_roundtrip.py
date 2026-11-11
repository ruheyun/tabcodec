import numpy as np
import pandas as pd
import pytest

from tabcodec.coders.onehot import OneHotCoder


@pytest.mark.parametrize("data", [
    ['a', 'b', 'a', 'c'],
    ['a', 'b', 'a', None, 'c'],
    [1, 2, 1, 3],
    ['only']
])
def test_onehot_roundtrip(data):
    s = pd.Series(data)
    coder = OneHotCoder().fit(s)
    back = coder.inverse_transform(coder.transform(s))
    pd.testing.assert_series_equal(
        back.reset_index(drop=True),
        s.reset_index(drop=True),
        check_names=False,
    )