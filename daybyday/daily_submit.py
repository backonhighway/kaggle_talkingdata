import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
LONG7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
LONG8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
LONG9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")
LONG_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")
#ERROR_ANALYSIS = os.path.join(OUTPUT_DIR, "bad_ip.csv")
PREDICTION1 = os.path.join(OUTPUT_DIR, "pred_day1_test.csv")
PREDICTION2 = os.path.join(OUTPUT_DIR, "pred_day2_test.csv")
PREDICTION3 = os.path.join(OUTPUT_DIR, "pred_day3_test.csv")


import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import column_selector
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()

train7 = pd.read_feather(LONG7)
train8 = pd.read_feather(LONG8)
train9 = pd.read_feather(LONG9)
test = pd.read_feather(LONG_TEST)
timer.time("load csv in ")


def train_it(train, pred_df, output_file):
    train_y = train["is_attributed"]
    train_x = train[predict_col]
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.1, random_state=99)
    timer.time("prepare train in ")

    lgb = pocket_lgb.GoldenLgb()
    model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
    lgb.show_feature_importance(model)
    del train, X_train, X_valid, y_train, y_valid
    gc.collect()
    timer.time("end train in ")

    submission = pd.DataFrame({"click_id": pred_df["click_id"].astype("int")})

    y_pred = model.predict(pred_df[predict_col])
    submission["is_attributed"] = y_pred
    print(submission.describe())
    timer.time("done prediction in ")

    submission.to_csv(output_file, index=False)
    timer.time("submission in ")

    return model


train_it(train7, test, PREDICTION1)
train_it(train8, test, PREDICTION2)
train_it(train9, test, PREDICTION3)

