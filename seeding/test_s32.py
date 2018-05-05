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
MODEL_FILE = os.path.join(FINAL_DIR, "pocket_final_model_test_s32.csv")
PREDICTION = os.path.join(FINAL_DIR, "pocket_final_pred_test_s32.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, seeding_test_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()

train7 = pd.read_feather(OUTPUT_DATA7)
train8 = pd.read_feather(OUTPUT_DATA8)
train9 = pd.read_feather(OUTPUT_DATA9)
test = pd.read_feather(OUTPUT_TEST)
test_old_id = dd.read_csv(OLD_INDEX).compute()
print(test_old_id.shape)
timer.time("load csv in ")

train = train7.append(train8).append(train9)
del train7, train8, train9
gc.collect()

y_train = train["is_attributed"]
X_train = train[predict_col]
timer.time("prepare train in ")

lgb = seeding_test_lgb.GoldenLgb(32)
model = lgb.do_train_sk(X_train, y_train)
lgb.show_feature_importance(model)
model.save_model(MODEL_FILE)
timer.time("end train in ")
del train, X_train, y_train
gc.collect()

test["old_click_id"] = np.array(test_old_id["old_click_id"])
test = test.sort_values("old_click_id")
timer.time("done sort in ")

y_pred = model.predict(test[predict_col])
print(y_pred.shape)
timer.time("done prediction in ")

mamas_idx = np.load(MAMAS_INDEX)
test = dd.read_csv(ORG_TEST).compute()
submission = pd.DataFrame({"click_id": test["click_id"].astype("int")})
submission["is_attributed"] = y_pred[mamas_idx]
print(submission.describe())
submission.to_csv(PREDICTION, index=False)
timer.time("submission in ")

