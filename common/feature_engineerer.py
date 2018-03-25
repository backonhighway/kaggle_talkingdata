import pandas as pd
import numpy as np


def do_feature_engineering(df):
    add_click_times(df)
    basic(df)
    #ip_to_cat(df)
    do_grouping(df)
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


def basic(df: pd.DataFrame):
    # at least one conversion?
    #df["ip_count"] = df.groupby("ip")["channel"].transform('count')
    df["app_count"] = df.groupby("app")["channel"].transform('count')
    #df["device_count"] = df.groupby("device")["channel"].transform('count')
    df["os_count"] = df.groupby("os")["channel"].transform('count')
    #df["channel_count"] = df.groupby("channel")["ip"].transform('count')
    #df["user_count"] = df.groupby(["ip", "device", "os"])["channel"].transform('count')
    df["hourly_click_count"] = df.groupby(["ip", "day", "hour"])["channel"].transform('count')
    #df["user_is_first_try"] = df.groupby(["ip", "app", "device", "os"])["channel"].shift(1)
    #df["user_is_first_try"] = np.where(df["user_is_first_try"].isnull(), 1, 0)
    df["idoa_is_last_try"] = df.groupby(["ip", "app", "device", "os"])["channel"].shift(-1)
    df["idoa_is_last_try"] = np.where(df["idoa_is_last_try"].isnull(), 1, 0)
    df["ido_is_last_try"] = df.groupby(["ip", "device", "os"])["channel"].shift(-1)
    df["ido_is_last_try"] = np.where(df["ido_is_last_try"].isnull(), 1, 0)


def add_click_times(df):
    df['day'] = df.click_time.str[8:10].astype(int)
    df['hour'] = df.click_time.str[11:13].astype(int)
    #df['time_min'] = df.click_time.str[14:16].astype(int)
    #df['time_sec'] = df.click_time.str[17:20].astype(int)
    df["click_time"] = pd.to_datetime(df["click_time"])


def do_grouping(df: pd.DataFrame):
    group_list = {
        "group_i": ["ip"],
        "group_ia": ["ip", "app"],
        "group_ido": ["ip", "device", "os"],
        "group_idoa": ["ip", "device", "os", "app"],
        "group_do": ["device", "os"],
    }
    for name, grouping in group_list.items():
        print(name)
        print(grouping)
        do_group_engineering(df, name, grouping)


def do_group_engineering(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    cnt_col = name + "_count"
    df[cnt_col] = grouper["channel"].transform("count")
    # fc_col = name + "_first_click"
    # lc_col = name + "_last_click"
    # df[fc_col] = grouper["channel"].shift(1)
    # df[fc_col] = np.where(df[fc_col].isnull(), 1, 0)
    # df[lc_col] = grouper["channel"].shift(-1)
    # df[lc_col] = np.where(df[lc_col].isnull(), 1, 0)

    if name not in ["group_ido", "group_i"]:
        return
    pct_col = name + "_prev_click_time"
    ict_col = name + "_interval_click_time"
    df[pct_col] = grouper["click_time"].shift(1)
    #df[pct_col] = df[pct_col].fillna()
    df[ict_col] = df["click_time"] - df[pct_col]
    df[ict_col] = df[ict_col].dt.total_seconds()
    df.drop(pct_col, axis=1, inplace=True)

    #min_col = name + "_min"
    max_col = name + "_max"
    std_col = name + "_std"
    mean_col = name + "_mean"
    sum_col = name + "_sum"
    #df[min_col] = df.groupby(grouping)[ict_col].transform("min")
    df[max_col] = df.groupby(grouping)[ict_col].transform("max")
    df[std_col] = df.groupby(grouping)[ict_col].transform("std")
    df[mean_col] = df.groupby(grouping)[ict_col].transform("mean")
    df[sum_col] = df.groupby(grouping)[ict_col].transform("sum")


def drop_unnecessary_col(df):
    drop_col = ["ip", "click_time", "day"]
    df.drop(drop_col, axis=1, inplace=True)
    if "attributed_time" in df.columns:
        df.drop("attributed_time", axis=1, inplace=True)


def get_necessary_col():
    use_col = [
        'app', 'device', 'os', 'channel', 'is_attributed',
        'hour', 'app_count', 'os_count',
        'idoa_is_last_try', "telling_ip"
        #'idoa_is_last_try', 'ioa_is_last_try', 'io_is_last_try',
        'group_i_count', 'group_ia_count', 'group_io_count', 'group_ic_count',
        'group_ioa_count', 'group_idoa_count', 'group_iac_count', 'group_ioc_count', 'group_ioac_count',
        'group_i_prev_click_time', 'group_i_next_click_time',
        #'group_io_prev_click_time', 'group_io_next_click_time',
        'group_ido_prev_click_time', 'group_ido_next_click_time',
        'group_i_nunique_os', 'group_i_nunique_app', 'group_i_nunique_channel',
        #'group_io_nunique_app', 'group_io_nunique_channel',
        'group_ido_nunique_app', 'group_ido_nunique_channel',
        #'group_ioct_max', 'group_ioct_std', 'group_ioct_mean', 'group_ioct_sum',
        'group_ict_max', 'group_ict_std', 'group_ict_mean', 'group_ict_sum',
        'group_idoct_max', 'group_idoct_std', 'group_idoct_mean', 'group_idoct_sum'
    ]
    return use_col


def get_test_col():
    use_col = [
        'app', 'device', 'os', 'channel',
        'hour', 'app_count', 'os_count',
        'idoa_is_last_try', "telling_ip",
        #'idoa_is_last_try', 'ioa_is_last_try', 'io_is_last_try',
        'group_i_count', 'group_ia_count', 'group_io_count', 'group_ic_count',
        'group_ioa_count', 'group_idoa_count', 'group_iac_count', 'group_ioc_count', 'group_ioac_count',
        'group_i_prev_click_time', 'group_i_next_click_time',
        #'group_io_prev_click_time', 'group_io_next_click_time',
        'group_ido_prev_click_time', 'group_ido_next_click_time',
        'group_i_nunique_os', 'group_i_nunique_app', 'group_i_nunique_channel',
        #'group_io_nunique_app', 'group_io_nunique_channel',
        'group_ido_nunique_app', 'group_ido_nunique_channel',
        #'group_ioct_max', 'group_ioct_std', 'group_ioct_mean', 'group_ioct_sum',
        'group_ict_max', 'group_ict_std', 'group_ict_mean', 'group_ict_sum',
        'group_idoct_max', 'group_idoct_std', 'group_idoct_mean', 'group_idoct_sum'
    ]
    return use_col
