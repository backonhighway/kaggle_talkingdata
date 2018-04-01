import pandas as pd

df1 = pd.DataFrame({
    "a": [1,2,3,3,4,4,5,5,4,3],
    "x": [1,2,3,4,5,6,7,8,9,10]
})
df2 = pd.DataFrame({
    "a": [1,2,3,3,5,5,4,3],
    "c": [21,22,23,24,25,26,27,28]
})

df1["rank"] = df1.groupby("a").rank().astype(int)
df2["rank"] = df2.groupby("a").rank().astype(int)

merged = pd.merge(df1, df2, on=["a","rank"], how="left")
print(merged)
exit(90)


nully = merged[merged["c"].isnull()]
nonully = merged[merged["c"].notnull()]
nonully = nonully.drop_duplicates(subset=["c"])
concated = nully.append(nonully)
concated.sort_index(inplace=True)
print(concated)

print("="*40)