import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
OUTPUT_CLUSTER = os.path.join(OUTPUT_DIR, "ip_cluster_day7.csv")

import pandas as pd
import numpy as np
import gc
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
train = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute()

timer.time("start click time")
train["click_time"] = pd.to_datetime(train["click_time"])
train["prev_click_time"] = train.groupby("ip")["click_time"].shift(1)
train["prev_click_time"] = train["click_time"] - train["prev_click_time"]
train["prev_click_time"] = train["prev_click_time"].dt.total_seconds()

timer.time("start counts")
g1 = train.groupby("ip")["device"].count().reset_index()
g2 = train.groupby("ip")["app"].nunique().reset_index()
g3 = train.groupby("ip")["device"].nunique().reset_index()
g4 = train.groupby("ip")["os"].nunique().reset_index()
g5 = train.groupby("ip")["prev_click_time"].std().reset_index()
g5["prev_click_time"] = g5["prev_click_time"].round()

gt = train.groupby("ip")["is_attributed"].mean().reset_index()

timer.time("start merge")
g1 = pd.merge(g1, g2, on="ip", how="left")
g1 = pd.merge(g1, g3, on="ip", how="left")
g1 = pd.merge(g1, g4, on="ip", how="left")
g1 = pd.merge(g1, g5, on="ip", how="left")
g1 = pd.merge(g1, gt, on="ip", how="left")
g1.columns = [
    "ip", "count", "nunique_app", "nunique_device", "nunique_os",
    "prev_click_time_std", "is_attributed_mean"
]

timer.time("doing output")
print(g1.head())
g1.to_csv(OUTPUT_CLUSTER, index=False)
