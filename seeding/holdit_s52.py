import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
FINAL_DIR = os.path.join(OUTPUT_DIR, "finals")
LONG7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
LONG8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
LONG9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
#ERROR_ANALYSIS = os.path.join(OUTPUT_DIR, "bad_ip.csv")
MODEL_FILE = os.path.join(FINAL_DIR, "pocket_final_model_val_s52.csv")
PREDICTION = os.path.join(FINAL_DIR, "pocket_final_pred_val_s52.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, holdout_validator2, seeding_val_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()

train7 = pd.read_feather(LONG7)
train8 = pd.read_feather(LONG8)
train9 = pd.read_feather(LONG9)
timer.time("load csv in ")

train = train7.append(train8)
holdout_df = train9
del train7, train8, train9
gc.collect()

y_train = train["is_attributed"]
X_train = train[predict_col]
y_valid = holdout_df["is_attributed"]
X_valid = holdout_df[predict_col]

timer.time("prepare train in ")
lgb = seeding_val_lgb.GoldenLgb(52)
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
model.save_model(MODEL_FILE)
lgb.show_feature_importance(model)

del train, X_train, X_valid, y_train, y_valid
gc.collect()
timer.time("end train in ")
validator = holdout_validator2.HoldoutValidator(model, holdout_df, predict_col)
validator.validate()
validator.output_prediction(PREDICTION)

timer.time("done validation in ")
