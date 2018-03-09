import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection

train = pd.read_csv('../input/train_first_1000k.csv')
# test = pd.read_csv('../input/test.csv')


#print(train.tail())
#print(train.head())
#print(train.describe())
#print(train.apply(pd.Series.nunique))

def add_count(df):
    df["ip_count"] = df.groupby("ip")["is_attributed"].transform('count')
    df["app_count"] = df.groupby("app")["is_attributed"].transform('count')
    df["device_count"] = df.groupby("device")["is_attributed"].transform('count')
    df["os_count"] = df.groupby("os")["is_attributed"].transform('count')
    df["channel_count"] = df.groupby("channel")["is_attributed"].transform('count')
    df["user_count"] = df.groupby(["ip", "device", "os"])["is_attributed"].transform('count')


add_count(train)
use_col = ["ip", "app", "device", "os", "channel", "is_attributed"]
drop_col = ["click_time", "attributed_time"]
train = train.drop(drop_col, axis=1)

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)


lgb = pocket_lgb.GoldenLgb()
lgb.do_train_sk(X_train, X_valid, y_train, y_valid)

print("end")
