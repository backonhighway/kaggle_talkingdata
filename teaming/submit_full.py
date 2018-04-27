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
TEST_DATA = os.path.join(OUTPUT_DIR, "short_test_merged.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sub_teaming.csv")

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
train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute()
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute()
test = dd.read_csv(TEST_DATA, dtype=dtypes).compute()
test = test[test["click_id"] >= 0]  # watch out this line for bugs
timer.time("load csv in ")

train7, train8, train9 = runtime_fe.get_additional_fe2(train7, train8, train9)
train7, train8, train9 = runtime_fe.get_prev_day_means(train7, train8, train9)
test = runtime_fe.get_additional_fe2_test(test)
test = runtime_fe.get_prev_day_mean_holdout(test, train9)
train = train7.append(train8).append(train9)
#train = runtime_fe.get_additional_fe(train)
#holdout_df = runtime_fe.get_additional_fe(holdout_df)
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
submission["is_attributed"] = y_pred
print(submission.describe())
#submission["is_attributed"] = submission["is_attributed"].rank(ascending=True)
timer.time("done prediction in ")

submission.to_csv(OUTPUT_FILE, index=False)
timer.time("submission in ")

