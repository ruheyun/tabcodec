from dataclasses import dataclass, field


@dataclass
class ColumnSpec:
    name: str
    kind: str = 'categorical'  # 'numeric' / 'categorical' / 'any'
    params: dict = field(default_factory=dict)