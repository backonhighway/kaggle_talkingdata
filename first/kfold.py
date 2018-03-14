import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection
import feature_engineerer
import holdout_validator

input_df = pd.read_csv('../input/train_day3_first_1000k.csv')

feature_engineerer.do_feature_engineering(input_df)
print(input_df.describe())

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

print("average score= ", total_score / split_number)

lgb.show_feature_importance(first_model)
validator = holdout_validator.HoldoutValidator(first_model)
validator.validate()