import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
FINAL_DIR = os.path.join(OUTPUT_DIR, "finals")
PREDICTION1 = os.path.join(FINAL_DIR, "pocket_final_pred_val_s99.csv")
PREDICTION2 = os.path.join(FINAL_DIR, "pocket_final_pred_val_s52.csv")
PREDICTION3 = os.path.join(FINAL_DIR, "pocket_final_pred_val_s54.csv")
LONG9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
PRED_OUT = os.path.join(FINAL_DIR, "final_val_seed_avg.csv")

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

pred1 = dd.read_csv(PREDICTION1, header=None).compute()
pred2 = dd.read_csv(PREDICTION2, header=None).compute()
pred3 = dd.read_csv(PREDICTION3, header=None).compute()
train9 = pd.read_feather(LONG9)
timer.time("load csv in ")


pred1 = np.array(pred1)
pred2 = np.array(pred2)
pred3 = np.array(pred3)
pred_avg = (pred1 + pred2 + pred3) / 3

y_true = train9["is_attributed"]
score = metrics.roc_auc_score(y_true, pred1)
print(score)
score = metrics.roc_auc_score(y_true, pred2)
print(score)
score = metrics.roc_auc_score(y_true, pred3)
print(score)
score = metrics.roc_auc_score(y_true, pred_avg)
print(score)
timer.time("done eval in ")

pred_out = pd.DataFrame(pred_avg)
pred_out.to_csv(PRED_OUT, index=False, header=False)
timer.time("done output in ")

