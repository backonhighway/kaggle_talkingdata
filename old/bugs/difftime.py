import pandas as pd
from dask import dataframe as dd

df1 = pd.DataFrame({
    "a": [1,1,3,3,2],
    "time":[
        "2018-01-01 10:00:00",
        "2018-01-01 10:10:00",
        "2018-01-01 10:10:03",
        "2018-01-01 10:23:00",
        "2018-01-01 11:32:00",
    ]
})

df1["time"] = pd.to_datetime(df1["time"])
df1["diff"] = df1["time"].diff(periods=1)
df1["seconds"] = df1["diff"].dt.seconds
df1["total"] = df1["diff"].dt.total_seconds()
print(df1)