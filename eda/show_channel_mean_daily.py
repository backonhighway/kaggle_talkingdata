import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv.org")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "channel_eda_daily.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader

print("started")
dtypes = csv_loader.get_dtypes()
df = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute()
grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"})
output_df = grouped.reset_index()
output_df["day"] = 7

df = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"})
output_df2 = grouped.reset_index()
output_df2["day"] = 8

df = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute()
grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"})
output_df3 = grouped.reset_index()
output_df3["day"] = 9

output_df = pd.merge(output_df, output_df2, on="channel", how="outer")
output_df = pd.merge(output_df, output_df3, on="channel", how="outer")
output_df.to_csv(OUTPUT_DATA, float_format='%.3f', index=False)
