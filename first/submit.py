import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection
import  feature_engineerer
import gc
import time

train = pd.read_csv('../input/train_day3_first_10000k.csv')

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

del train
gc.collect()

s_start_time = time.time()
test = pd.read_csv('../input/test.csv')
feature_engineerer.do_feature_engineering(test)
y_pred = model.predict(test)
submission = pd.DataFrame({"click_id": test["click_id"], "is_attributed": y_pred})
print(submission.describe())
print("done prediction")
submission.to_csv("../output/submission.csv", float_format='%.6f', index=False)


# sample = pd.read_csv('../input/sample_submission.csv', usecols=["click_id"])
# output = sample.merge(submission, on="click_id", how="left")
# output["is_attributed"] = output["is_attributed"].fillna(1)
# print(output.describe())
#
# output.to_csv("../output/only_public.csv", float_format='%.6f', index=False)
#
s_end_time = time.time()
print("submission in ", s_end_time - s_start_time)

