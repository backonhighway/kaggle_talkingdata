import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
OUTPUT_DATA7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
OUTPUT_DATA8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
OUTPUT_DATA9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")
OUTPUT_CLUSTER = os.path.join(OUTPUT_DIR, "ip_cluster.csv")

import pandas as pd
import numpy as np
import gc
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()

train7 = pd.read_feather(OUTPUT_DATA7)
train8 = pd.read_feather(OUTPUT_DATA8)
train9 = pd.read_feather(OUTPUT_DATA9)
train7["day"] = 7
train8["day"] = 8
train9["day"] = 9
train = train7.append(train8).append(train9)

timer.time("start click time")
train["click_time"] = pd.to_datetime(train["click_time"])
train["prev_click_time"] = train.groupby("ip")["click_time"].shift(1)
train["prev_click_time"] = train["click_time"] - train["prev_click_time"]
train["prev_click_time"] = train["prev_click_time"].dt.total_seconds()

timer.time("start counts")
group_col = ["ip", "day"]
g1 = train.groupby(group_col)["device"].count().reset_index()
g2 = train.groupby(group_col)["app"].nunique().reset_index()
g3 = train.groupby(group_col)["device"].nunique().reset_index()
g4 = train.groupby(group_col)["os"].nunique().reset_index()
g5 = train.groupby(group_col)["prev_click_time"].std().reset_index()
g5["prev_click_time"] = g5["prev_click_time"].round()

gt = train.groupby(group_col)["is_attributed"].mean().reset_index()

timer.time("start merge")
g1 = pd.merge(g1, g2, on=group_col, how="left")
g1 = pd.merge(g1, g3, on=group_col, how="left")
g1 = pd.merge(g1, g4, on=group_col, how="left")
g1 = pd.merge(g1, g5, on=group_col, how="left")
g1 = pd.merge(g1, gt, on=group_col, how="left")
g1.columns = [
    "ip", "days", "count", "nunique_app", "nunique_device", "nunique_os",
    "prev_click_time_std", "is_attributed_mean"
]

timer.time("doing output")
print(g1.head())
g1.to_csv(OUTPUT_CLUSTER, index=False)
