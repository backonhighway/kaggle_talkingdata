import pandas as pd
from dask import dataframe as dd

df1 = pd.DataFrame({
    "a": [1,2,3,3,4,4,5,5,4,3],
    "v": [1,1,3,4,3,10,4,10,10,10],
})

s = df1.groupby(["a"])["v"].size()
print(s)

r = s.reset_index()
print(r)
