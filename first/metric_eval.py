import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
TRAIN_DATA = os.path.join(INPUT_DIR, "train.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection, metrics
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb

dtypes = csv_loader.get_dtypes()
num_row = 1000 * 1000 * 1
train = pd.read_csv(TRAIN_DATA, dtype=dtypes, nrows=num_row)

feature_engineerer.do_feature_engineering(train)
print(train.describe())

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)

print(model.best_score)
y_pred = model.predict(X_valid)
y_true = y_valid
score = metrics.roc_auc_score(y_true, y_pred)
print(score)

y_pred = model.predict(train_x)
y_true = train["is_attributed"]
score = metrics.roc_auc_score(y_true, y_pred)
print(score)

