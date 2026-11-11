import numpy as np
import pandas as pd

from ..base import BaseCoder


_NAN = object()


class OrdinalCoder(BaseCoder):
    def __init__(self):
        self.categories_ = []