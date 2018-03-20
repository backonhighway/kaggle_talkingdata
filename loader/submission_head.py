import pandas as pd
import gc
import csv_loader

file_name = "../output/submission.csv"
dtypes = csv_loader.get_dtypes()

df = pd.read_csv(file_name, nrows=1000*1000*1)
print("done loading...")


df.to_csv("../output/submission_head.csv", index=False)


