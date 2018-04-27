import pandas as pd

df1 = pd.DataFrame({
    "a": [1,2,3,3,4,4,5,5,4,3],
    "x": [1,2,3,4,5,6,7,8,9,10]
})
df2 = pd.DataFrame({
    "a": [1,2,3,3,5,5,3],
    "c": [21,22,23,24,25,26,28]
})


print(df1.iloc[1:3])

