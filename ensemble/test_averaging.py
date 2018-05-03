import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
LONG7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
LONG8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
LONG9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
PREDICTION1 = os.path.join(OUTPUT_DIR, "pred_day1_testm.csv")
PREDICTION2 = os.path.join(OUTPUT_DIR, "pred_day2_testm.csv")
PREDICTION3 = os.path.join(OUTPUT_DIR, "pred_day3_testm.csv")
PREDICTION_ALL = os.path.join(OUTPUT_DIR, "005_test.csv")
PRED_OUT = os.path.join(OUTPUT_DIR, "daily_avg.csv")

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

pred7 = dd.read_csv(PREDICTION1).compute()
pred8 = dd.read_csv(PREDICTION2).compute()
pred9 = dd.read_csv(PREDICTION3).compute()
pred7 = np.array(pred7["is_attributed"])
pred8 = np.array(pred8["is_attributed"])
pred9 = np.array(pred9["is_attributed"])
pred_all = dd.read_csv(PREDICTION_ALL).compute()
pred_all_array = np.array(pred_all["is_attributed"])
timer.time("load csv in ")

pred_daily = (pred_all_array + (pred7 + pred8 + pred9) / 3) / 2
pred_all["is_attributed"] = pred_daily
timer.time("done setup in ")
pred_all.to_csv(PRED_OUT, index=False)
timer.time("done output in ")

