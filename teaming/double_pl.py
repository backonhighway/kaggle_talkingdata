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
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import runtime_fe, column_selector
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()

train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute().sample(n=1000*1000*1, random_state=99)
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute().sample(n=1000*1000*1, random_state=99)
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute().sample(n=1000*1000*1, random_state=99)
timer.time("load csv in ")

timer.time("start runtime_fe")
train7, train8, train9 = runtime_fe.get_prev_day_mean(train7, train8, train9)
train = train7.append(train8)
holdout_df = train9
print(train["hour"].describe())
timer.time("done runtime fe")

predict_col = column_selector.get_predict_col()
train_y = train["is_attributed"]
train_x = train[predict_col]
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

timer.time("prepare train in ")
lgb = pocket_lgb.get_eval_lgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)

y_pred = model.predict(holdout_df[predict_col])

timer.time("end train in ")
validator = holdout_validator2.HoldoutValidator(model, holdout_df, predict_col)
validator.validate()
validator.validate_rmse(ERROR_ANALYSIS)
#validator.output_prediction(PREDICTION)
timer.time("done validation in ")

####################
# second round
####################

pl_data = holdout_df[predict_col].copy()
pl_data["pseudo_label"] = y_pred
# pl_label = pl_label[1]  # change to series
#pl_data = pl_data.sample(n=1000*1000*1)

X_train = X_train.append(pl_data)
y_train = y_train.append(pl_data["pseudo_label"])

timer.time("prepare train2 in ")
lgb = pocket_lgb.get_eval_lgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)

timer.time("end train2 in ")
validator = holdout_validator2.HoldoutValidator(model, holdout_df, predict_col)
validator.validate()
validator.validate_rmse(ERROR_ANALYSIS)
timer.time("done validation2 in ")
