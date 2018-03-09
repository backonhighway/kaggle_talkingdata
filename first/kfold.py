import pandas as pd
import numpy as np
import pocket_lgb
from sklearn import model_selection

input_df = pd.read_csv('../input/train_first_1000k.csv')
# test = pd.read_csv('../input/test.csv')


#print(train.tail())
#print(train.head())
#print(train.describe())
#print(train.apply(pd.Series.nunique))


def add_count(df):
    df["ip_count"] = df.groupby("ip")["is_attributed"].transform('count')
    df["device_count"] = df.groupby("device")["is_attributed"].transform('count')
    df["os_count"] = df.groupby("os")["is_attributed"].transform('count')
    df["user_count"] = df.groupby(["ip", "device", "os"])["is_attributed"].transform('count')
    df["app_count"] = df.groupby("app")["is_attributed"].transform('count')
    df["channel_count"] = df.groupby("channel")["is_attributed"].transform('count')


def target_encode(df):
    df["ip_target"] = df.groupby("ip")["is_attributed"].transform(np.mean)
    df["device_target"] = df.groupby("device")["is_attributed"].transform(np.mean)
    df["os_target"] = df.groupby("os")["is_attributed"].transform(np.mean)
    df["user_target"] = df.groupby(["ip", "device", "os"])["is_attributed"].transform(np.mean)
    df["app_target"] = df.groupby("app")["is_attributed"].transform(np.mean)
    df["channel_target"] = df.groupby("channel")["is_attributed"].transform(np.mean)


use_col = ["ip", "app", "device", "os", "channel", "is_attributed"]
drop_col = ["click_time", "attributed_time"]
input_df = input_df.drop(drop_col, axis=1)
add_count(input_df)
# train_y = input_df["is_attributed"]
# train_x = input_df.drop("is_attributed", axis=1)

skf = model_selection.KFold(n_splits=5)
# for train_index, test_index in skf.split(input_df, input_df["is_attributed"]):
#    train = input_df[train_index]
#    test = input_df[test_index]
for train_index, test_index in skf.split(input_df):
    train_np = input_df.iloc[train_index]
    test_np = input_df.iloc[test_index]
    train_df = pd.DataFrame(train_np)
    test_df = pd.DataFrame(test_np)

    #t_t = input_df.iloc(train_index)
    #te_t = input_df.iloc(test_index)

    lgb = pocket_lgb.GoldenLgb()
    lgb.do_train(train_df, test_df)
    #lgb.do_train_sk(X_train, X_test, y_train, y_test)



print("end")
