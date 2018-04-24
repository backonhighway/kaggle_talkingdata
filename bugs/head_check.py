import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PL_DATA = os.path.join(OUTPUT_DIR, "pseudo_with_mean_encode.csv")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
ERROR_ANALYSIS = os.path.join(OUTPUT_DIR, "bad_ip.csv")
PREDICTION = os.path.join(OUTPUT_DIR, "pocket_prediction.csv")

import pandas as pd
import numpy as np
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()

train = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute()
print(train.head())

