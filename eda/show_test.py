import pandas as pd
import csv_loader

file_name = "../input/test.csv"
dtypes = csv_loader.get_dtypes()

df = pd.read_csv(file_name, dtype=dtypes, nrows=1000*100*1)

# df["index1"] = df.index
# print(df.head())
#
# not_same_df = df[df["index1"] != df["click_id"]]
# print(not_same_df.head())
print(df.tail())
df.to_csv("../output/test_100k.csv")