import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection

train = pd.read_csv('../input/train_sample.csv')
# test = pd.read_csv('../input/test.csv')


#print(train.head())
#print(train.describe())
#print(train.apply(pd.Series.nunique))


train["ip_count"] = train.groupby("ip")["is_attributed"].transform('count')
train["ip_target"] = train.groupby("ip")["is_attributed"].transform(np.mean)

use_col = ["ip", "app", "device", "os", "channel", "is_attributed"]
train = train[use_col]
train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)


lgb = pocket_lgb.GoldenLgb()
lgb.do_train_sk(X_train, X_valid, y_train, y_valid)

print("end")
