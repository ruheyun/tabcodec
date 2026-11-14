import pandas as pd
from tabcodec import ColumnSpec, TabCodec


df = pd.DataFrame({
    'age': [25, 30, 22, 40],
    'city': ['beijing', 'shanghai', 'beijing', 'shenzhen'],
    'score': [0.5, 1.2, 0.8, 2.0],
})

spec = [
    ColumnSpec('age', kind='numeric', coder='minmax'),
    ColumnSpec('city', kind='categorical', coder='ordinal'),
    ColumnSpec('score', kind='numeric', coder='standard'),
]


codec = TabCodec(spec, output='frame').fit(df)
Z = codec.transform(df)
back = codec.inverse_transform(Z)

print(Z)
print(Z.shape, codec.output_dim)
print(back)

