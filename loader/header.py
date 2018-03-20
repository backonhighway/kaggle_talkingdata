import pandas as pd
import gc
import csv_loader

file_name = "../input/train_day4.csv"
dtypes = csv_loader.get_dtypes()

df = pd.read_csv(file_name, dtype=dtypes, nrows=1000*100*1)
print(df.head())

df.to_csv("../output/train_d4_100k.csv", index=False)

file_name = "../input/test_old.csv"
dtypes = csv_loader.get_dtypes()

df = pd.read_csv(file_name, dtype=dtypes, nrows=1000*100*1)
print("done loading...")


df.to_csv("../output/train_old_100k.csv", index=False)


