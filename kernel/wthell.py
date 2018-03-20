import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb


path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

import feature_engineerer
import gc

train_df = pd.read_csv('../input/train.csv', nrows=1000000, dtype=dtypes)
feature_engineerer.do_feature_engineering(train_df)
len_train = len(train_df)

val_df = train_df[(len_train-300000):len_train]
train_df = train_df[:(len_train-300000)]
target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'hourly_click_count']
categorical = ['app','device','os', 'channel', 'hour']


print("Training...")

import pocket_lgb
trainer = pocket_lgb.GoldenLgb()
bst = trainer.do_train(train_df, val_df)
del train_df
del val_df
gc.collect()

# import holdout_validator
# validator = holdout_validator.HoldoutValidator(bst)
# validator.validate()
# exit(0)

test = pd.read_csv('../input/test.csv', dtype=dtypes)
feature_engineerer.do_feature_engineering(test)
print(test.describe())
y_pred = bst.predict(test[predictors])
submission = pd.DataFrame({"click_id": test["click_id"], "is_attributed": y_pred})

submission.to_csv('../output/for_compare.csv',index=False)
print("done...")
print(submission.info())
