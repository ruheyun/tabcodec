import pandas as pd
from src.tabcodec.coders.onehot import OneHotCoder


s = pd.Series(['a', 'b', 'a', None, 'c'])
coder = OneHotCoder().fit(s)
Z = coder.transform(s)
back = coder.inverse_transform(Z)

print(Z.shape, coder.output_dim)
print(back.tolist())
print(Z)

