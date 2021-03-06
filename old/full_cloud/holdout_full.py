import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(OUTPUT_DIR, "full_train_day3_featured.csv")
HOLDOUT_DATA = os.path.join(OUTPUT_DIR, "full_train_day4_featured.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
train = pd.read_csv(TRAIN_DATA, dtype=dtypes)
print(train.info())

train = train[feature_engineerer.get_necessary_col()]
train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)
timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)

del train, X_train, X_valid, y_train, y_valid
gc.collect()
timer.time("end train in ")
validator = holdout_validator.HoldoutValidator(model, HOLDOUT_DATA)
validator.validate()

timer.time("done validation in ")
