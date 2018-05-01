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

DIV_FILE7 = os.path.join(OUTPUT_DIR, "div_train7.csv")
DIV_FILE8 = os.path.join(OUTPUT_DIR, "div_train8.csv")
DIV_FILE9 = os.path.join(OUTPUT_DIR, "div_train9.csv")
DIV_HOLDOUT = os.path.join(OUTPUT_DIR, "div_hold.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, pocket_lgb, pocket_timer, pocket_logger
from talkingdata.divident import div_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()
train7 = pd.read_feather(OUTPUT_DATA9)
# train8 = pd.read_feather(OUTPUT_DATA8)
# train9 = pd.read_feather(OUTPUT_DATA9)
# test = pd.read_feather(OUTPUT_TEST)
# test = test[test["click_id"] >= 0]  # watch out this line for bugs
timer.time("load csv in ")

train7["start_time"] = pd.to_datetime("2017-11-09 00:00:00")
train7["click_time"] = pd.to_datetime(train7["click_time"]) + pd.DateOffset(hours=8)
train7["hour"] = train7["click_time"].dt.hour
print(train7[["start_time", "click_time", "hour"]].tail())
timer.time("done prep")

df_list = []
window_size = 4
for start_hour in range(0, 22, 4):
    print(start_hour)
    end_hour = start_hour + window_size
    mask = (train7["hour"] >= start_hour) & (train7["hour"] < end_hour)
    tmp_df = train7[mask].copy()
    if tmp_df is None:
        continue
    tmp_df["start_hour"] = start_hour
    tmp_df["end_hour"] = end_hour
    df_list.append(div_fe.get_features(tmp_df, start_hour, window_size))

result_df = pd.concat(df_list)
timer.time("got df")

div_col = column_selector.get_div_col()
train_col = div_col + ["ip", "click_time", "is_attributed"]
result_df[train_col].reset_index(drop=True).to_feather(DIV_HOLDOUT)
timer.time("output df")
