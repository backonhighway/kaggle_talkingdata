import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection

train = pd.read_csv('../input/train_first_1000k.csv')

#print(train.tail())
#print(train.head())
#print(train.describe())
#print(train.apply(pd.Series.nunique))


def add_count(df):
    df["ip_count"] = df.groupby("ip")["channel"].transform('count')
    df["app_count"] = df.groupby("app")["channel"].transform('count')
    df["device_count"] = df.groupby("device")["channel"].transform('count')
    df["os_count"] = df.groupby("os")["channel"].transform('count')
    df["channel_count"] = df.groupby("channel")["ip"].transform('count')
    df["user_count"] = df.groupby(["ip", "device", "os"])["channel"].transform('count')
    df["kernel_qty"] = df.groupby(["ip", "time_day", "time_hour"])["channel"].transform('count')


def click_times(df):
    df['time_day'] = df.click_time.str[8:10]
    df['time_hour'] = df.click_time.str[11:13]
    df['time_min'] = df.click_time.str[14:16]
    df['time_sec'] = df.click_time.str[17:20]



add_count(train)
use_col = ["ip", "app", "device", "os", "channel", "is_attributed"]
drop_col = ["click_time", "attributed_time"]
train = train.drop(drop_col, axis=1)

train_y = train["is_attributed"]
train_x = train.drop("is_attributed", axis=1)

X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)


lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)
print("end")
# clicks in last x time
# is last click of user of the app?

