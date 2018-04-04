import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(INPUT_DIR, "train.csv")
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "channel_eda.csv")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer

dtypes = csv_loader.get_dtypes()
train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes, usecols=["ip"])
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes, usecols=["ip"])
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes, usecols=["ip"])
print(train7["ip"].nunique())
print(train8["ip"].nunique())
print(train9["ip"].nunique())

train7 = train7[train7["ip"] <= 126420]
train8 = train8[train8["ip"] <= 126420]
train9 = train9[train9["ip"] <= 126420]
print(train7["ip"].nunique())
print(train8["ip"].nunique())
print(train9["ip"].nunique())