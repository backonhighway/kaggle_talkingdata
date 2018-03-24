import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(OUTPUT_DIR, "train_day3_featured.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer


timer = pocket_timer.GoldenTimer()
dtypes = csv_loader.get_featured_dtypes()
#num_row = 1000 * 100
#input_df = pd.read_csv(TRAIN_DATA, dtype=dtypes, nrows=num_row)
input_df = pd.read_csv(TRAIN_DATA, dtype=dtypes)
print(input_df.info())
input_df = input_df[feature_engineerer.get_necessary_col()]

split_number = 5
skf = model_selection.KFold(n_splits=split_number)
lgb = pocket_lgb.GoldenLgb()
first_model = None
total_score = 0
for train_index, test_index in skf.split(input_df):
    train_np = input_df.iloc[train_index]
    test_np = input_df.iloc[test_index]
    train_df = pd.DataFrame(train_np)
    test_df = pd.DataFrame(test_np)

    model = lgb.do_train(train_df, test_df)
    score = model.best_score["valid_0"]["auc"]
    total_score += score

    if first_model is None:
        first_model = model
    else:
        del model
        gc.collect()

print("average score= ", total_score / split_number)

lgb.show_feature_importance(first_model)
exit(0)
validator = holdout_validator.HoldoutValidator(first_model)
validator.validate()
