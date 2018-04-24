import pandas as pd
from sklearn import model_selection
import numpy as np
from concurrent import futures


def get_oof_ch_mean(df: pd.DataFrame):
    kf = model_selection.KFold(n_splits=8, random_state=99)
    ret_list = []
    for fold_index, oof_index in kf.split(df):
        #df.iloc[oof_index] = get_channel_mean(df.iloc[fold_index], df.iloc[oof_index])
        ret_list.append(get_channel_mean(df.iloc[fold_index], df.iloc[oof_index]))
    ret_df = pd.DataFrame(pd.concat(ret_list))
    return ret_df


def get_channel_mean(fold_df: pd.DataFrame, oof_df: pd.DataFrame):
    grouped = fold_df.groupby("channel")["is_attributed"].agg({"mean", "count"}).reset_index()
    grouped.columns = ["channel", "ch_mean", "ch_count"]
    grouped["ch_log_count"] = np.log1p(grouped["ch_count"])
    grouped["weighted_ch_mean"] = grouped["ch_mean"] * grouped["ch_log_count"]

    ret_df = pd.merge(oof_df, grouped, on="channel", how="left")
    ret_df["ip_ch_mean"] = ret_df.groupby("ip")["ch_mean"].transform("mean")
    ret_df["ip_ch_count"] = ret_df.groupby("ip")["ch_count"].transform("mean")

    weighted_ch_mean_series = ret_df.groupby("ip")["weighted_ch_mean"].transform("sum")
    weighted_ch_log_count_series = ret_df.groupby("ip")["ch_log_count"].transform("sum")
    ret_df["ip_weighted_ch_mean"] = weighted_ch_mean_series / weighted_ch_log_count_series

    # grouped2 = fold_df.groupby("ip")["is_attributed"].agg({"mean", "count"}).reset_index()
    # grouped2.columns = ["ip", "ip_mean", "ip_count"]
    # ret_df = pd.merge(ret_df, grouped2, on="ip", how="left")

    # ret_df["app_ch_mean"] = ret_df.groupby("app")["mean"].transform("mean")
    # ret_df["app_ch_count"] = ret_df.groupby("app")["count"].transform("mean")
    # ret_df["os_ch_mean"] = ret_df.groupby("os")["mean"].transform("mean")
    # ret_df["os_ch_count"] = ret_df.groupby("os")["count"].transform("mean")
    return ret_df


def get_holdout_channel_mean(df: pd.DataFrame, holdout_df: pd.DataFrame):
    grouped = df.groupby("channel")["is_attributed"].agg({"mean", "count"}).reset_index()
    grouped.columns = ["channel", "ch_mean", "ch_count"]
    grouped["ch_log_count"] = np.log1p(grouped["ch_count"])
    grouped["weighted_ch_mean"] = grouped["ch_mean"] * grouped["ch_log_count"]

    holdout_df = pd.merge(holdout_df, grouped, on="channel", how="left")
    holdout_df["ip_ch_mean"] = holdout_df.groupby("ip")["ch_mean"].transform("mean")
    holdout_df["ip_ch_count"] = holdout_df.groupby("ip")["ch_count"].transform("mean")
    weighted_ch_mean_series = holdout_df.groupby("ip")["weighted_ch_mean"].transform("sum")
    weighted_ch_log_count_series = holdout_df.groupby("ip")["ch_log_count"].transform("sum")
    holdout_df["ip_weighted_ch_mean"] = weighted_ch_mean_series / weighted_ch_log_count_series

    # holdout_df["app_ch_mean"] = holdout_df.groupby("app")["mean"].transform("mean")
    # holdout_df["app_ch_count"] = holdout_df.groupby("app")["count"].transform("mean")
    # holdout_df["os_ch_mean"] = holdout_df.groupby("os")["mean"].transform("mean")
    # holdout_df["os_ch_count"] = holdout_df.groupby("os")["count"].transform("mean")

    # grouped2 = df.groupby("ip")["is_attributed"].agg({"mean", "count"}).reset_index()
    # grouped2.columns = ["ip", "ip_mean", "ip_count"]
    # holdout_df = pd.merge(holdout_df, grouped2, on="ip", how="left")
    return holdout_df


def get_digit(number, n):
    return number // 10**n % 10


def get_additional_fe(df: pd.DataFrame):
    #df["ip_1"] = df["ip"].apply(lambda ip: get_digit(ip, 0))
    #df["ip_2"] = df["ip"].apply(lambda ip: get_digit(ip, 1))
    #df["ip_12"] = (df["ip_1"] * 10) + df["ip_2"]

    #df["hourly_ip_ch_mean"] = df.groupby(["ip", "hour"])["ip_ch_mean"].transform("mean")
    #df["hourly_ip_ch_count"] = df.groupby(["ip", "hour"])["ip_ch_count"].transform("mean")
    return df


# LAG feature, top nunique
def get_prev_day_mean(df_day1, df_day2, df_day3):
    grouped = df_day1.groupby("ip")["is_attributed"].mean().reset_index()
    grouped.columns = ["ip", "ip_prev_day_mean_encoding"]
    df_day2 = pd.merge(df_day2, grouped, on="ip", how="left")
    df_day1["ip_prev_day_mean_encoding"] = np.NaN

    grouped = df_day2.groupby("ip")["is_attributed"].mean().reset_index()
    grouped.columns = ["ip", "ip_prev_day_mean_encoding"]
    df_day3 = pd.merge(df_day3, grouped, on="ip", how="left")

    return df_day1, df_day2, df_day3


def get_prev_day_means(df_day1, df_day2, df_day3):
    col1 = "ip_prev_day_mean_encoding"
    col2 = "ido_prev_day_mean_encoding"
    col3 = "idoa_prev_day_mean_encoding"
    g1 = ["ip"]
    g2 = ["ip", "device", "os"]
    g3 = ["ip", "device", "os", "app"]

    df_day1, df_day2, df_day3 = get_prev_day_mean1(df_day1, df_day2, df_day3, col1, g1)
    df_day1, df_day2, df_day3 = get_prev_day_mean1(df_day1, df_day2, df_day3, col2, g2)
    df_day1, df_day2, df_day3 = get_prev_day_mean1(df_day1, df_day2, df_day3, col3, g3)
    return df_day1, df_day2, df_day3


def get_prev_day_mean1(df_day1, df_day2, df_day3, col_name, group_col):
    grouped = df_day1.groupby(group_col)["is_attributed"].mean().reset_index()
    col_list = list(group_col) + list(col_name)
    grouped.columns = col_list
    df_day2 = pd.merge(df_day2, grouped, on=group_col, how="left")
    df_day1[col_name] = np.NaN

    grouped = df_day2.groupby(group_col)["is_attributed"].mean().reset_index()
    grouped.columns = col_list
    df_day3 = pd.merge(df_day3, grouped, on=group_col, how="left")

    return df_day1, df_day2, df_day3


def get_prev_day_mean_all(df_day1, df_day2, df_day3):
    # BOTU
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_list = list()
        col1 = "ip_prev_day_mean_encoding"
        col2 = "ido_prev_day_mean_encoding"
        col3 = "idoa_prev_day_mean_encoding"
        g1 = ["ip"]
        g2 = ["ip", "device", "os"]
        g3 = ["ip", "device", "os", "app"]
        future_list.append(executor.submit(get_prev_day_mean1, df_day1, df_day2, df_day3, col1, g1))
        future_list.append(executor.submit(get_prev_day_mean1, df_day1, df_day2, df_day3, col2, g2))
        future_list.append(executor.submit(get_prev_day_mean1, df_day1, df_day2, df_day3, col3, g3))







