import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
TEST_DATA = os.path.join(OUTPUT_DIR, "short_merged_test_vanilla.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission_merged_full.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

use_col = feature_engineerer.get_necessary_col()
dtypes = csv_loader.get_featured_dtypes()
train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes, usecols=use_col).head(10000)
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes, usecols=use_col).head(10000)
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes, usecols=use_col).head(10000)
train = train7.append(train8).append(train9)
print(train.info())
del train7, train8, train9
gc.collect()
timer.time("load csv in ")

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)
timer.time("prepare train in ")

lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)
timer.time("end train in ")
del train, X_train, X_valid, y_train, y_valid
gc.collect()

use_col = feature_engineerer.get_submit_col()
org_test = dd.read_csv(ORG_TEST, dtype=dtypes).compute()
print(org_test.info())
test = dd.read_csv(TEST_DATA, dtype=dtypes, usecols=use_col).compute()
test = pd.merge(org_test, test, on="click_id", how="left")
test = test.drop_duplicates(subset=['click_id'])
print(test.info())
submission = pd.DataFrame({"click_id": test["click_id"]})
test = test.drop("click_id", axis=1)
exit(0)

y_pred = model.predict(test)
submission["is_attributed"] = y_pred
print(submission.describe())
#submission["is_attributed"] = submission["is_attributed"].rank(ascending=True)
timer.time("done prediction in ")

submission.to_csv(OUTPUT_FILE, index=False)
timer.time("submission in ")

