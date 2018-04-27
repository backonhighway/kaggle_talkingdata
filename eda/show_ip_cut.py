import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
TEST_DATA = os.path.join(OUTPUT_DIR, "short_test_merged.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import runtime_fe, column_selector
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute()
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute()
test = dd.read_csv(TEST_DATA, dtype=dtypes).compute()
timer.time("load csv in ")

print(train7.groupby("hour")["ip"].describe())
print(train8.groupby("hour")["ip"].describe())
print(train9.groupby("hour")["ip"].describe())
print(test.groupby("hour")["ip"].describe())

exit(0)
train7 = train7[train7["hour"] < 23]
train8 = train8[train8["hour"] < 23]
train9 = train9[train9["hour"] < 23]

max7 = train7["ip"].max().compute()
max8 = train8["ip"].max().compute()
max9 = train9["ip"].max().compute()
print(max7)
print(max8)
print(max9)
train8 = train8[train8["ip"] > max7]
train9 = train9[train9["ip"] > max8]
min7 = train7["ip"].min().compute()
min8 = train8["ip"].min().compute()
min9 = train9["ip"].min().compute()
print(min7)
print(min8)
print(min9)

max_test = test["ip"].max().compute()
print(max_test)
