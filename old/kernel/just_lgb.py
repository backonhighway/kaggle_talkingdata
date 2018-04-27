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
train = pd.read_csv('../input/train.csv', nrows=1000000, dtype=dtypes)

feature_engineerer.do_feature_engineering(train)
print(train.describe())

len_train = len(train)

val_df = train[(len_train-300000):len_train]
train_df = train[:(len_train-300000)]

#train_y = train["is_attributed"]
#train_x = train.drop("is_attributed", axis=1)

#X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)
timer.time("prepare train in ")

lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train(train_df, val_df)
#model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
#lgb.show_feature_importance(model)
timer.time("end train in ")
del train_df
del val_df
gc.collect()
predictors = ['app','device','os', 'channel', 'hour', 'hourly_click_count']
categorical = ['app','device','os', 'channel', 'hour']


test = pd.read_csv('../input/test.csv', dtype=dtypes)
feature_engineerer.do_feature_engineering(test)
print(test.describe())
y_pred = model.predict(test[predictors])
submission = pd.DataFrame({"click_id": test["click_id"], "is_attributed": y_pred})

submission.to_csv('../output/for_compare2.csv',index=False)
print("done...")
print(submission.info())


# test = pd.read_csv('../input/test.csv', dtype=dtypes)
# feature_engineerer.do_feature_engineering(test)
# print(test.describe())
# y_pred = model.predict(test)
# submission = pd.DataFrame({"click_id": test["click_id"], "is_attributed": y_pred})
# print(submission.describe())
# #submission["is_attributed"] = submission["is_attributed"].rank(ascending=True)
# print("done prediction")
# submission.to_csv("../output/submission2.csv", index=False)
#
#
# # sample = pd.read_csv('../input/sample_submission.csv', usecols=["click_id"])
# # output = sample.merge(submission, on="click_id", how="left")
# # output["is_attributed"] = output["is_attributed"].fillna(1)
# # print(output.describe())
# #
# # output.to_csv("../output/only_public.csv", float_format='%.6f', index=False)
# #
# timer.time("submission in ")

