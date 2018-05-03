import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
OUTPUT_DATA7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
OUTPUT_DATA8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
OUTPUT_DATA9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")
MAMAS_INDEX = os.path.join(INPUT_DIR, "last_test_idx.npy")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sub_long.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector, runtime_fe
from talkingdata.common import csv_loader, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()
train7 = pd.read_feather(OUTPUT_DATA7)
train8 = pd.read_feather(OUTPUT_DATA8)
train9 = pd.read_feather(OUTPUT_DATA9)
test = pd.read_feather(OUTPUT_TEST)
test = test[test["click_id"] >= 0]  # watch out this line for bugs
timer.time("load csv in ")

train = train7.append(train8).append(train9)
print(train.info())
print(test.info())
del train7, train8, train9
gc.collect()
timer.time("done runtime fe")

train_y = train["is_attributed"]
train_x = train[predict_col]
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)
timer.time("prepare train in ")

lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)
timer.time("end train in ")
del train, X_train, X_valid, y_train, y_valid
gc.collect()


submission = pd.DataFrame({"click_id": test["click_id"].astype("int")})

y_pred = model.predict(test[predict_col])
mamas_idx = np.load(MAMAS_INDEX)
submission["is_attributed"] = y_pred[mamas_idx]
print(submission.describe())
timer.time("done prediction in ")

submission.to_csv(OUTPUT_FILE, index=False)
timer.time("submission in ")

