import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(INPUT_DIR, "train.csv")
TRAIN_DATA8 = os.path.join(INPUT_DIR, "train_day8.csv")
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "channel_eda_ip.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer

timer = pocket_timer.GoldenTimer()
timer.time("started")

dtypes = csv_loader.get_dtypes()
# df = pd.read_csv(TRAIN_DATA, dtype=dtypes, nrows=1000*100)
df = dd.read_csv(TRAIN_DATA, dtype=dtypes).compute()
df['hour'] = df.click_time.str[11:13].astype(int)

# df["first"] = df.groupby("ip")["hour"].transform("first")
# print(df)
# exit(0)

firsty = df.groupby("ip")["hour"].first().reset_index()
target_mean = df.groupby("ip")["is_attributed"].mean().reset_index()
print(firsty.head())
print(target_mean.head())
merged = pd.merge(firsty, target_mean, on="ip", how="outer")
print(merged.head())
groupy = merged.groupby("hour")["is_attributed"].mean().reset_index()
print(groupy)

