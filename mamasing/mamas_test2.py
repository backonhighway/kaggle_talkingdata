import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
FINAL_DIR = os.path.join(OUTPUT_DIR, "finals")
OUTPUT_DATA7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
OUTPUT_DATA8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
OUTPUT_DATA9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")

MAMAS_INDEX = os.path.join(INPUT_DIR, "last_test_idx.npy")
OLD_INDEX = os.path.join(INPUT_DIR, "test_index.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")

MODEL_FILE = os.path.join(FINAL_DIR, "pocket_final_model_test_s99.csv")
PREDICTION = os.path.join(FINAL_DIR, "pocket_final_pred_test_s99.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()

test = pd.read_feather(OUTPUT_TEST)
test_old_id = dd.read_csv(OLD_INDEX).compute()
print(test_old_id.shape)
timer.time("load csv in ")

test["old_click_id"] = np.array(test_old_id["old_click_id"])
test = test.sort_values("old_click_id")
timer.time("done sort in ")

import lightgbm as lgb
model = lgb.Booster(model_file=MODEL_FILE)
y_pred = model.predict(test[predict_col])
print(y_pred.shape)
timer.time("done prediction in ")

mamas_idx = np.load(MAMAS_INDEX)
test = dd.read_csv(ORG_TEST).compute()
submission = pd.DataFrame({"click_id": test["click_id"].astype("int")})
submission["is_attributed"] = y_pred[mamas_idx]
#submission = submission[submission["click_id"] >= 0]  # watch out this line for bugs
print(submission.describe())
submission.to_csv(PREDICTION, index=False)
timer.time("submission in ")

