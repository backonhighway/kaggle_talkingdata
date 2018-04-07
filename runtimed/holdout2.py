import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA89 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import runtime_fe
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
train9 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
#print(train.info())
timer.time("load csv in ")

holdout_df = train9[train9["hour"] >= 12]
train9 = [train9["hour"] < 12]
train = train8.append(train9)

# timer.time("start runtime_fe")
# runtime_fe.get_oof_ch_mean(train)
# timer.time("got ch mean")
# holdout_df = runtime_fe.get_holdout_channel_mean(train, holdout_df)
# timer.time("got holdout ch mean")
print(train.info())
print(holdout_df.info())

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)

del train, X_train, X_valid, y_train, y_valid
gc.collect()
timer.time("end train in ")
validator = holdout_validator2.HoldoutValidator(model, holdout_df)
validator.validate()

timer.time("done validation in ")