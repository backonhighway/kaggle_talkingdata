import pandas as pd


df1 = pd.read_csv('../output/kernel_edited_sub.csv', nrows=1000)
df2 = pd.read_csv('../output/submission.csv', nrows=1000)

print(df1.head(10))

print(df2.head(10))