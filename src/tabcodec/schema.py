from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    name: str
    kind: str = 'categorical'  # 'numeric' / 'categorical' / 'any'
    coder: str = 'onehot'
    params: dict = field(default_factory=dict)