import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
TRAIN_DATA = os.path.join(INPUT_DIR, "train_day9.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
use_col = feature_engineerer.get_necessary_col()
dtypes = csv_loader.get_dtypes()
train = dd.read_csv(TRAIN_DATA, dtype=dtypes).compute()
#print(train.info())
timer.time("load csv in ")

merge_col = ["ip", "app", "device", "os", "channel", "day", "hour", "time_min", "time_sec"]
train['day'] = train.click_time.str[8:10].astype(int)
train['hour'] = train.click_time.str[11:13].astype(int)
train['time_min'] = train.click_time.str[14:16].astype(int)
train['time_sec'] = train.click_time.str[17:20].astype(int)

train["rank"] = train.groupby(merge_col).rank().astype("int")

print(train.groupby("rank")["is_attributed"].mean())
