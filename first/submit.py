import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection
import feature_engineerer
import gc
import time
import csv_loader
import pocket_timer

timer = pocket_timer.GoldenTimer()
dtypes = csv_loader.get_dtypes()
train = pd.read_csv('../input/train.csv', nrows=10000000, dtype=dtypes)

feature_engineerer.do_feature_engineering(train)
print(train.describe())

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2)
timer.time("prepare train in ")

lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)
timer.time("end train in ")
del train, X_train, X_valid, y_train, y_valid
gc.collect()

s_start_time = time.time()
test = pd.read_csv('../input/test.csv', dtype=dtypes)
submission = pd.DataFrame({"click_id": test["click_id"]})
feature_engineerer.do_feature_engineering(test)
print(test.describe())
test = test.drop("click_id", axis=1)
y_pred = model.predict(test)
submission["is_attributed"] = y_pred
print(submission.describe())
#submission["is_attributed"] = submission["is_attributed"].rank(ascending=True)
print("done prediction")
submission.to_csv("../output/submission.csv", index=False)


# sample = pd.read_csv('../input/sample_submission.csv', usecols=["click_id"])
# output = sample.merge(submission, on="click_id", how="left")
# output["is_attributed"] = output["is_attributed"].fillna(1)
# print(output.describe())
#
# output.to_csv("../output/only_public.csv", float_format='%.6f', index=False)
#
timer.time("submission in ")

