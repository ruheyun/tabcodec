# tabcodec

面向混合类型表格数据的编码 / 反编码工具箱。为每一列指定一种编码方式，`transform` 输出模型可用的数值矩阵，`inverse_transform` 把编码结果还原成原始表格。

## 特性

- **混合列类型**：数值列与类别列可混用，按 `ColumnSpec` 自动分发。
- **双向可逆**：每种编码器实现 `fit / transform / inverse_transform`，编码后能还原。
- **声明式配置**：用 `ColumnSpec` 描述每列怎么编码，配置可序列化、可复现。
- **显式缺失值策略**：数值列默认拒绝 NaN 并给出出错行号；类别列将 NaN 视为独立类别。

## 安装
最新版本
```bash
uv add "git+https://github.com/ruheyun/tabcodec"
```
或者
```bash
uv pip install "git+https://github.com/ruheyun/tabcodec"
```

特定版本
```bash
uv add "git+https://github.com/ruheyun/tabcodec@0.1.0"
```
或者
```bash
uv pip install "git+https://github.com/ruheyun/tabcodec@v0.1.0"
```

## 快速开始

```python
import pandas as pd
from tabcodec import ColumnSpec, TabCodec

df = pd.DataFrame({
    "age":    [25, 30, 35, 40],
    "city":   ["beijing", "shanghai", "beijing", "shenzhen"],
    "score":  [0.5, 1.2, 0.8, 2.0],
})

spec = [
    ColumnSpec("age",   kind="numeric",     coder="standard"),
    ColumnSpec("city",  kind="categorical", coder="onehot"),
    ColumnSpec("score", kind="numeric",     coder="standard"),
]

codec = TabCodec(spec, output="frame").fit(df)
Z = codec.transform(df)
print(Z)
print("output_dim =", codec.output_dim)

df_back = codec.inverse_transform(Z)
print(df_back)
```

`output="frame"` 返回带列名的 DataFrame（多列编码会展开成 `city_0`、`city_1` …）；`output="array"` 返回纯 numpy 矩阵，适合直接喂模型。


## 当前支持的编码器

| 名称 | 类 | 适用 | 说明 |
|---|---|---|---|
| `onehot` | `OneHotCoder` | 类别列 | 支持 `drop=None/"first"`；未知类别编码为全 0，反编码为 NaN |
| `ordinal` | `OrdinalCoder` | 类别列 | 按类别首次出现顺序映射为整数；未知类别为 -1，反编码为 NaN |
| `standard` | `StandardScalerCoder` | 数值列 | 标准化；默认拒绝 NaN |
| `passthrough` | `PassthroughCoder` | 任意 | 原样透传 |

### 数值列的缺失值

数值编码器默认 `allow_nan=False`，`fit` 和 `transform` 阶段检测到 NaN 会直接抛 `ValueError`，并给出出错行号：

```python
ValueError: Numeric column contains NaN at row(s) [2]. Clean the data
before encoding, or pass allow_nan=True to keep NaN values (not suitable
for neural networks).
```

如需保留 NaN（调试、对比实验、树模型场景），显式声明：

```python
ColumnSpec("age", kind="numeric", coder="standard", params={"allow_nan": True})
```

注意：`allow_nan=True` 时编码结果含 NaN，**不能直接输入神经网络**。

### 类别列的缺失值

类别编码器把 NaN 视为一个独立类别，拥有自己的 one-hot 列 / 序号，`inverse_transform` 能精确还原为 NaN。这与数值列的策略刻意不对称：类别缺失本身可能携带信息，而数值缺失通常需要先清洗。

## 反编码的语义

- `onehot` / `ordinal` / `standard` / `passthrough` 在数据合法时**严格可逆**（浮点误差除外）。
- 未知类别（训练时未见过的值）无法还原，`inverse_transform` 返回 NaN，而不是静默映射到某个已有类别。
- 未来加入的 `embedding` 类编码器属于**近似可逆**，反编码走最近邻查表，会在文档中单独标注。


## 路线图

- [x] `onehot` / `ordinal` / `standard` / `passthrough`
- [ ] `analog_bits`：含非 2 的幂类别数下的边界处理（参考 ResBit 思路）
- [ ] `embedding`：高基数类别与文本列，标注为近似可逆
- [ ] `TabCodec` 支持保存 / 加载（`save` / `load`）以复现编码配置
- [ ] 编码结果的列名映射导出

## License
MIT LICENSE
