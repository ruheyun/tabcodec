from abc import ABC, abstractmethod


class BaseCoder(ABC):
    """所有表格编码器的统一接口
    transform: 原始一列  -->  编码矩阵(num_samples, output_dim)
    inverse_transform: 编码矩阵  -->  原始一列(pd.Series)
    """

    is_exact_inverse: bool = True

    @abstractmethod
    def fit(self, X, y=None) -> "BaseCoder":
        ...

    @abstractmethod
    def transform(self, X):
        ...

    @abstractmethod
    def inverse_transform(self, Z):
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

        