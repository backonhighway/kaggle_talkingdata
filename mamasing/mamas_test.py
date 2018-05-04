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
MODEL_FILE = os.path.join(FINAL_DIR, "pocket_final_model_test_s99.csv")
PREDICTION = os.path.join(OUTPUT_DIR, "pocket_final_test_s99.csv")

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

train7 = pd.read_feather(OUTPUT_DATA7).head(1000*1000)
train8 = pd.read_feather(OUTPUT_DATA8).head(1000*1000)
train9 = pd.read_feather(OUTPUT_DATA9).head(1000*1000)
test = pd.read_feather(OUTPUT_TEST)
test_old_id = dd.read_csv(OLD_INDEX).compute()
timer.time("load csv in ")

train = train7.append(train8).append(train9)
print(train.info())
print(test.info())
del train7, train8, train9
gc.collect()

y_train = train["is_attributed"]
X_train = train[predict_col]
timer.time("prepare train in ")

lgb = mamas_test_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, y_train)
lgb.show_feature_importance(model)
timer.time("end train in ")
del train, X_train, y_train
gc.collect()

test["old_click_id"] = np.array(test_old_id["old_click_id"])
test = test.sort_values("old_click_id")
timer.time("done sort in ")

submission = pd.DataFrame({"click_id": test["click_id"].astype("int")})
test = test[test["click_id"] >= 0]
y_pred = model.predict(test[predict_col])
mamas_idx = np.load(MAMAS_INDEX)
submission["is_attributed"] = y_pred[mamas_idx]
submission = submission[submission["click_id"] >= 0]  # watch out this line for bugs
print(submission.describe())
timer.time("done prediction in ")
exit(0)
submission.to_csv(PREDICTION, index=False)
timer.time("submission in ")

