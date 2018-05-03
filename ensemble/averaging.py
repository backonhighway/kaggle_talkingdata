import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
LONG7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
LONG8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
LONG9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
PREDICTION1 = os.path.join(OUTPUT_DIR, "pred_day1_val.csv")
PREDICTION2 = os.path.join(OUTPUT_DIR, "pred_day2_val.csv")
PREDICTION_ALL = os.path.join(OUTPUT_DIR, "005_val.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import metrics
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()

pred7 = dd.read_csv(PREDICTION1, header=None).compute()
pred8 = dd.read_csv(PREDICTION2, header=None).compute()
pred7 = np.array(pred7)
pred8 = np.array(pred8)
train9 = pd.read_feather(LONG9)
pred_all = dd.read_csv(PREDICTION_ALL, header=None).compute()
pred_all = np.array(pred_all)
timer.time("load csv in ")

y_true = train9["is_attributed"]
y_pred = pred_all + ((pred7 + pred8) / 2) / 2
score = metrics.roc_auc_score(y_true, y_pred)
print(score)

# 0.9831090041924851
# y_pred = pred7
# score = metrics.roc_auc_score(y_true, y_pred)
# print(score)

# 0.9831638783891637
# y_pred = pred8
# score = metrics.roc_auc_score(y_true, y_pred)
# print(score)

# 0.9836784545953107
# y_pred = (pred7 + pred8) / 2
# score = metrics.roc_auc_score(y_true, y_pred)
# print(score)

timer.time("done ez in ")
# 0.9838554019924455
# y_pred = pred_all
# score = metrics.roc_auc_score(y_true, y_pred)
# print(score)

# 0.9838156490677148
# y_pred = (pred_all + pred8)/2
# score = metrics.roc_auc_score(y_true, y_pred)
# print(score)

timer.time("done scoring in ")