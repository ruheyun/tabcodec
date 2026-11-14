from .coders.onehot import OneHotCoder
from .coders.ordinal import OrdinalCoder
from .coders.numeric import StandardScalerCoder
from .coders.passthrough import PassthroughCoder


_REGISTRY = {
    'onehot': OneHotCoder,
    'ordinal': OrdinalCoder,
    'standard': StandardScalerCoder,
    'passthrough': PassthroughCoder,
}


def create_coder(name: str, **params):
    if name not in _REGISTRY:
        raise KeyError(f'Unknown coder: {name!r}. Available: {list(_REGISTRY)}')
    return _REGISTRY[name](**params)

def register(name: str, cls):
    _REGISTRY[name] = cls

    