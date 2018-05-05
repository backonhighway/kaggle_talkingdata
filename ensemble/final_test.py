import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
FINAL_DIR = os.path.join(OUTPUT_DIR, "finals")
PREDICTION1 = os.path.join(FINAL_DIR, "pocket_final_pred_test_s99.csv")
PREDICTION2 = os.path.join(FINAL_DIR, "pocket_final_pred_test_s32.csv")
PREDICTION3 = os.path.join(FINAL_DIR, "pocket_final_pred_test_s52.csv")
PREDICTION4 = os.path.join(FINAL_DIR, "pocket_final_pred_test_s54.csv")
PREDICTION5 = os.path.join(FINAL_DIR, "pocket_final_pred_test_s92.csv")
PRED_OUT = os.path.join(FINAL_DIR, "final_test_seed_avg.csv")

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

pred1 = dd.read_csv(PREDICTION1).compute()
pred2 = dd.read_csv(PREDICTION2).compute()
pred3 = dd.read_csv(PREDICTION3).compute()
pred4 = dd.read_csv(PREDICTION4).compute()
pred5 = dd.read_csv(PREDICTION5).compute()
timer.time("load csv in ")

print(pred1.head())
print(pred2.head())
print(pred3.head())
print(pred4.head())
print(pred5.head())
pred_out = pd.DataFrame()
pred_out["click_id"] = np.array(pred1["click_id"]).astype("int")

pred1 = np.array(pred1["is_attributed"])
pred2 = np.array(pred2["is_attributed"])
pred3 = np.array(pred3["is_attributed"])
pred4 = np.array(pred4["is_attributed"])
pred5 = np.array(pred5["is_attributed"])
pred_avg = (pred1 + pred2 + pred3 + pred4 + pred5) / 5

pred_out["is_attributed"] = pred_avg
print(pred_out.head())
print(pred_out.describe())
timer.time("done setup in ")

pred_out.to_csv(PRED_OUT, index=False)
timer.time("done output in ")

