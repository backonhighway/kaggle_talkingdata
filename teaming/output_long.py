import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
TEST_DATA = os.path.join(OUTPUT_DIR, "short_test_merged.csv")
OUTPUT_DATA7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
OUTPUT_DATA8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
OUTPUT_DATA9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import runtime_fe, column_selector
from talkingdata.common import csv_loader, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()

# train7 = pd.read_csv(TRAIN_DATA7, dtype=dtypes, nrows=1000*100)
# train8 = pd.read_csv(TRAIN_DATA8, dtype=dtypes, nrows=1000*100)
# train9 = pd.read_csv(TRAIN_DATA9, dtype=dtypes, nrows=1000*100)
# test = pd.read_csv(TEST_DATA, dtype=dtypes, nrows=1000*100)
train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute()
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute()
test = dd.read_csv(TEST_DATA, dtype=dtypes).compute()
timer.time("load csv in ")

#holdout_df = train9[train9["hour"] >= 8]
#train9 = train9[train9["hour"] < 8]
#train8 = train8[train8["hour"] >= 8]
#train = train8.append(train9)

timer.time("start runtime_fe")
train7, train8, train9 = runtime_fe.get_additional_fe2(train7, train8, train9)
train7, train8, train9 = runtime_fe.get_prev_day_means(train7, train8, train9)
test = runtime_fe.get_additional_fe2_test(test)
test = runtime_fe.get_prev_day_mean_holdout(test, train9)
timer.time("done runtime fe")

predict_col = column_selector.get_predict_col()
train7[predict_col].reset_index(drop=True).to_feather(OUTPUT_DATA7)
train8[predict_col].reset_index(drop=True).to_feather(OUTPUT_DATA8)
train9[predict_col].reset_index(drop=True).to_feather(OUTPUT_DATA9)
test[predict_col].reset_index(drop=True).to_feather(OUTPUT_TEST)
timer.time("done output")
