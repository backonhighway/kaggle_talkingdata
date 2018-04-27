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

train7, train8, train9 = runtime_fe.get_additional_fe2(train7, train8, train9)
timer.time("got ip_cat ")


def show_mean(df, group_col, the_logger):
    the_mean = df.groupby(group_col)["is_attributed"].mean()
    print(the_mean)
    the_logger.info(the_mean)


for train in train7, train8, train9:
    show_mean(train, "ip_cat", logger)

all_df = train7.append(train8).append(train9)
show_mean(all_df, "ip_cat1", logger)
show_mean(all_df, "ip_cat2", logger)
show_mean(all_df, "ip_cat3", logger)

timer.time("done show")