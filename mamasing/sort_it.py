import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
FINAL_DIR = os.path.join(OUTPUT_DIR, "finals")

OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")
SORTED_TEST = os.path.join(OUTPUT_DIR, "sorted_test.feather")
MAMAS_INDEX = os.path.join(INPUT_DIR, "last_test_idx.npy")
OLD_INDEX = os.path.join(INPUT_DIR, "test_index.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, mamas_test_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()

test = pd.read_feather(OUTPUT_TEST)
test_old_id = dd.read_csv(OLD_INDEX).compute()
timer.time("load csv in ")

test["old_click_id"] = np.array(test_old_id["old_click_id"])
test = test.sort_values("old_click_id")
timer.time("done sort in ")

test.reset_index(drop=True).to_feather(SORTED_TEST)
timer.time("output sort in ")

