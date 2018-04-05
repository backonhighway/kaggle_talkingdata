import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(INPUT_DIR, "train.csv")
TRAIN_DATA8 = os.path.join(INPUT_DIR, "short_train_day8.csv")
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "channel_eda_ip.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader

print("started")
dtypes = csv_loader.get_dtypes()
df = dd.read_csv(TRAIN_DATA, dtype=dtypes).compute()
grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"})
output_df = grouped.reset_index()

df8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()

df = pd.merge(df8, df, on="channel", how="left")
ip_attr = df.groupby("ip")["is_attributed"].agg({"mean", "count"}).reset_index()
ip_channel_mean = df.groupbe("ip")["mean"].agg({"mean"}).reset_index()
df = pd.merge(ip_attr, ip_channel_mean, on="ip", how="left")

output_df.to_csv(OUTPUT_DATA, float_format='%.6f', index=False)
