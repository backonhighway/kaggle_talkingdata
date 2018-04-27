import pandas as pd
from dask import dataframe as dd

df1 = pd.DataFrame({
    "a": [1,2,3,3,4,4,5,5,4,3],
    "x": [1,2,3,4,5,6,7,8,9,10],
})


df1["diff"] = df1["a"].diff(periods=1)
print(df1)

print("="*40)

df = dd.from_pandas(df1, npartitions=2)
df["diff2"] = df["a"].diff(periods=1)
print(df.compute())

df1["grouped"] = df1.groupby("a")["x"].diff(periods=1)
print(df1)

ret = df.groupby("a")["x"].get_group("a").repartition(npartitions=1)
df["test"] = ret.diff(periods=1).compute()
print(df)
#df1["grouped"] = df.groupby("a")["x"].get_group("a").diff(periods=1)