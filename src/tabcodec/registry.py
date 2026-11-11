from .coders.onehot import OneHotCoder


_REGISTRY = {
    'onehot': OneHotCoder,
}


def create_coder(name: str, **params):
    if name not in _REGISTRY:
        raise KeyError(f'Unknown coder: {name!r}. Available: {list(_REGISTRY)}')
    return _REGISTRY[name](**params)

def register(name: str, cls):
    _REGISTRY[name] = cls

    