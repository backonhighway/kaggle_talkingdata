import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(OUTPUT_DIR, "train_day3_featured.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb

dtypes = csv_loader.get_featured_dtypes()
num_row = 1000 * 100
train = pd.read_csv(TRAIN_DATA, dtype=dtypes, nrows=num_row)
print(train.info())
print(train.describe())

train = train[feature_engineerer.get_necessary_col()]

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)


lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)
# clicks in last x time
# is last click of user of the app?

validator = holdout_validator.HoldoutValidator(model)
validator.validate()
