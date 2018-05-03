import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PREDICTION1 = os.path.join(OUTPUT_DIR, "pred_day1_test.csv")
PREDICTION2 = os.path.join(OUTPUT_DIR, "pred_day2_test.csv")
PREDICTION3 = os.path.join(OUTPUT_DIR, "pred_day3_test.csv")
PREDICTION1m = os.path.join(OUTPUT_DIR, "pred_day1_testm.csv")
PREDICTION2m = os.path.join(OUTPUT_DIR, "pred_day2_testm.csv")
PREDICTION3m = os.path.join(OUTPUT_DIR, "pred_day3_testm.csv")


import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector, runtime_fe
from talkingdata.common import csv_loader, pocket_lgb, pocket_timer, pocket_logger

test = dd.read_csv(PREDICTION2).compute()
test = test[test["click_id"] >= 0]  # watch out this line for bugs
test.to_csv(PREDICTION2m, index=False)