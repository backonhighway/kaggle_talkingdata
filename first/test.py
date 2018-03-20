import pandas as pd
import numpy as np
import pocket_lgb
import holdout_validator
import feature_engineerer
from sklearn import model_selection
import csv_loader

dtypes = csv_loader.get_dtypes()
num_row = 1000 * 1000 * 10
train = pd.read_csv('../input/train.csv', dtype=dtypes, nrows=num_row)

feature_engineerer.do_feature_engineering(train)
print(train.describe())

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

