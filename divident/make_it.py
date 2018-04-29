import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
OUTPUT_DATA7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
OUTPUT_DATA8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
OUTPUT_DATA9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sub_long.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector, runtime_fe
from talkingdata.common import csv_loader, pocket_lgb, pocket_timer, pocket_logger
import pytz

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()
train7 = pd.read_feather(OUTPUT_DATA7).head(1000*100)
# train8 = pd.read_feather(OUTPUT_DATA8)
# train9 = pd.read_feather(OUTPUT_DATA9)
# test = pd.read_feather(OUTPUT_TEST)
# test = test[test["click_id"] >= 0]  # watch out this line for bugs
timer.time("load csv in ")

df_list = []
window_size = 4
for start_hour in range(0, 4, 2):
    print(start_hour)
    end_hour = start_hour + window_size
    mask = (train7["hour"] >= start_hour) & (train7["hour"] < end_hour)
    tmp_df = train7[mask].copy()
    tmp_df["click_time"] = pd.to_datetime(tmp_df["click_time"]) + pd.DateOffset(hours=8)
    tmp_df["start_time"] = pd.to_datetime("2017-11-07 00:00:00")
    tmp_df["start_time"] = tmp_df["start_time"] + pd.DateOffset(hours=start_hour)
    tmp_df["end_time"] = tmp_df["start_time"] +  pd.DateOffset(hours=window_size)
    tmp_df["time_till_start"] = tmp_df["click_time"] - tmp_df["start_time"]
    tmp_df["time_till_start"] = tmp_df["time_till_start"].dt.total_seconds()
    tmp_df["time_till_end"] = tmp_df["end_time"] - tmp_df["click_time"]
    tmp_df["time_till_end"] = tmp_df["time_till_end"].dt.total_seconds()

    done_col = ["click_time", "start_time", "end_time", "time_till_start", "time_till_end"]
    print(tmp_df[done_col].head())


