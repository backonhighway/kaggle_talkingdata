import pandas as pd
from dask import dataframe as dd

df1 = pd.DataFrame({
    "a": [3,4,3,3,4,4,5,5,4,3],
    "v": [10,10,3,3,3,10,4,10,10,10],
})

s = df1.groupby(["a"])["v"].value_counts()
print(s)
r = s.groupby("a").nlargest(1)
print(r)
ri = r.reset_index(drop=True)
print(ri)

l = df1.groupby(["a"])["v"].transform(lambda x: x.value_counts().nlargest(1))
print(l)