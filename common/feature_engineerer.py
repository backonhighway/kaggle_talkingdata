import pandas as pd
import numpy as np


def do_feature_engineering(df):
    add_click_times(df)
    add_count(df)
    ip_to_cat(df)
    drop_unnecessary_col(df)


def get_digit(number, n):
    return number // 10**n % 10


def ip_to_cat(df):
    df["ip_1"] = df["ip"].apply(lambda ip: get_digit(ip, 0))
    df["ip_2"] = df["ip"].apply(lambda ip: get_digit(ip, 1))
    df["ip_3"] = df["ip"].apply(lambda ip: get_digit(ip, 2))
    df["ip_4"] = df["ip"].apply(lambda ip: get_digit(ip, 3))
    df["ip_5"] = df["ip"].apply(lambda ip: get_digit(ip, 4))
    df["ip_6"] = df["ip"].apply(lambda ip: get_digit(ip, 5))


def add_count(df: pd.DataFrame):
    #df["ip_count"] = df.groupby("ip")["channel"].transform('count')
    #df["app_count"] = df.groupby("app")["channel"].transform('count')
    #df["device_count"] = df.groupby("device")["channel"].transform('count')
    #df["os_count"] = df.groupby("os")["channel"].transform('count')
    #df["channel_count"] = df.groupby("channel")["ip"].transform('count')
    df["user_count"] = df.groupby(["ip", "device", "os"])["channel"].transform('count')
    df["hourly_click_count"] = df.groupby(["ip", "time_day", "time_hour"])["channel"].transform('count')
    df["user_is_first_try"] = df.groupby(["ip", "app", "device", "os"])["channel"].shift(1)
    df["user_is_first_try"] = np.where(df["user_is_first_try"].isnull(), 1, 0)
    df["user_is_last_try"] = df.groupby(["ip", "app", "device", "os"])["channel"].shift(-1)
    df["user_is_last_try"] = np.where(df["user_is_last_try"].isnull(), 1, 0)


def add_click_times(df):
    df['time_day'] = df.click_time.str[8:10]
    df['time_hour'] = df.click_time.str[11:13]
    df['time_min'] = df.click_time.str[14:16]
    df['time_sec'] = df.click_time.str[17:20]


def drop_unnecessary_col(df):
    drop_col = ["click_time", "time_day", "time_hour", "time_min", "time_sec"]
    df.drop(drop_col, axis=1, inplace=True)
    if "attributed_time" in df.columns:
        df.drop("attributed_time", axis=1, inplace=True)
