from .coders.onehot import OneHotCoder
from .coders.ordinal import OrdinalCoder
from .coders.numeric import StandardScalerCoder


_REGISTRY = {
    'onehot': OneHotCoder,
    'ordinal': OrdinalCoder,
    'standard': StandardScalerCoder,
}


def create_coder(name: str, **params):
    if name not in _REGISTRY:
        raise KeyError(f'Unknown coder: {name!r}. Available: {list(_REGISTRY)}')
    return _REGISTRY[name](**params)

def register(name: str, cls):
    _REGISTRY[name] = cls

    