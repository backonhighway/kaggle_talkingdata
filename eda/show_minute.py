import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA8 = os.path.join(INPUT_DIR, "train_day8.csv")
OUTPUT_DATA_MIN = os.path.join(OUTPUT_DIR, "min_eda.csv")
OUTPUT_DATA_SEC = os.path.join(OUTPUT_DIR, "sec_eda.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer

timer = pocket_timer.GoldenTimer()
timer.time("started")

dtypes = csv_loader.get_dtypes()
df = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
timer.time("read first")

df["click_time"] = pd.to_datetime(df["click_time"])
df["minute"] = df["click_time"].dt.minute
df["second"] = df["click_time"].dt.second
timer.time("done fe")

min_df = df.groupby("minute")["is_attributed"].mean().reset_index()
min_df.to_csv(OUTPUT_DATA_MIN, index=False)

sec_df = df.groupby("second")["is_attributed"].mean().reset_index()
sec_df.to_csv(OUTPUT_DATA_SEC, index=False)
timer.time("end")