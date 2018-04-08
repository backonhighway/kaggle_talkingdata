import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

from concurrent import futures
import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer


timer = pocket_timer.GoldenTimer()


def get_time(df: dd.DataFrame):
    #df['day'] = df.click_time.str[8:10].astype(int)
    df["click_time"] = dd.to_datetime(df["click_time"])
    timer.time("done click time")
    df["hour"] = df["click_time"].dt.hour
    #df["telling_ip"] = np.where(df["ip"] <= 126420, 1, 0)


def get_last_try(df: pd.DataFrame):
    col_name = "idoa_is_last_try"
    series = df.groupby(["ip", "app", "device", "os"])["channel"].shift(-1)
    series = np.where(series.isnull(), 1, 0)
    return [(col_name, series)]


def get_first_appear_hour(df: pd.DataFrame):
    col_name = "first_appear_hour"
    series = df.groupby("ip")["hour"].transform("first")
    return [(col_name, series)]


def submit_tasks(df, executor):
    future_list = []
    group_list = {
        "group_ido": ["ip", "device", "os"],
        "group_idoa": ["ip", "app", "os", "device"],
        "group_ioac": ["ip", "app", "os", "channel"],
        "group_idoac": ["ip", "app", "os", "channel", "device"],
    }
    for name, grouping in group_list.items():
        future_list.append(executor.submit(get_counts, df, name, grouping))
        future_list.append(executor.submit(get_interval_click_time, df, name, grouping))

    n_unique_list = {
        "group_i": ["ip"]
    }
    for name, grouping in n_unique_list.items():
        future_list.append(executor.submit(get_count_share, df, name, grouping))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "os"))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "app"))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "channel"))
        future_list.append(executor.submit(get_click_interval_and_stats, df, name, grouping))

    stats_list = {
        "group_ido": ["ip", "device", "os"],
    }
    for name, grouping in stats_list.items():
        future_list.append(executor.submit(get_click_interval_and_stats, df, name, grouping))

    all_group_list = {
        "group_idoac": ["ip", "device", "os", "channel", "app"]
    }
    #for name, grouping in all_group_list.items():
    #    get_interval_click_time(df, name, grouping)

    return future_list


def get_count_share(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    cnt_col = name + "_count"
    count_series = grouper["device"].transform("count")

    grouping.append("hour")
    grouper = df.groupby(grouping)
    hour_col = name + "_hourly_count"
    hour_series = grouper["device"].transform("count")
    share_col = name + "_hourly_count_share"
    share_series = hour_series / count_series
    return [(cnt_col, count_series), (hour_col, hour_series), (share_col, share_series)]


def get_counts(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    cnt_col = name + "_count"
    series = grouper["device"].transform("count")
    return [(cnt_col, series)]


def get_nunique(df: pd.DataFrame, name: str, grouping:list, nunique_col:str):
    grouper = df.groupby(grouping)
    uni_col = name + "_nunique"+ "_" + nunique_col
    series = grouper[nunique_col].transform("nunique")
    return [(uni_col, series)]


def get_click_interval_and_stats(df: pd.DataFrame, name: str, grouping:list):
    ret_list = []
    ret_list.extend(get_interval_click_time(df, name, grouping))
    ret_list.extend(get_short_stats(df, name, grouping, ret_list[0]))
    return ret_list


def get_interval_click_time(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    pct_col = name + "_prev_click_time"
    prev_series = grouper["click_time"].shift(1)
    prev_series = df["click_time"] - prev_series
    prev_series = prev_series.dt.total_seconds()

    nct_col = name + "_next_click_time"
    next_series = grouper["click_time"].shift(-1)
    next_series = next_series - df["click_time"]
    next_series = next_series.dt.total_seconds()

    return [(pct_col, prev_series), (nct_col, next_series)]


def get_short_stats(df: pd.DataFrame, name: str, grouping: list, pct_tuple: tuple):
    pct_col = pct_tuple[0]
    df[pct_col] = pct_tuple[1]
    sum_col = name + "_ct_sum"
    sum_series = df.groupby(grouping)[pct_col].transform("sum")
    std_col = name + "_ct_std"
    std_series = df.groupby(grouping)[pct_col].transform("std")
    return [(sum_col, sum_series), (std_col, std_series)]


def get_rolling_mean(df: pd.DataFrame, name: str, grouping:list):
    df["channel_mean"] = df.groupby("channel")["is_attributed"].transform("mean")
    # maybe shift the mean... do different windows size, groupby ido too
    mean_func = lambda x: x.rolling(window=10).mean()
    df["ip_mean"] = df.groupby("ip")["channel_mean"].transform(mean_func)



def make_file(input_file, output_file):
    dtypes = csv_loader.get_dtypes()
    input_df = dd.read_csv(input_file, dtype=dtypes).repartition(npartitions=32)
    timer.time("load csv")
    get_time(input_df)
    timer.time("got time")
    input_df = input_df.compute()
    timer.time("got pandas dataframe")

    with futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_list = list()
        future_list.append(executor.submit(get_last_try, input_df))
        # future_list.extend(executor.submit(get_first_appear_hour, input_df))
        future_list.extend(submit_tasks(input_df, executor))
    timer.time("done executor")

    for one_future in future_list:
        list_of_tuples = one_future.result()
        for results in list_of_tuples:
            col_name, series = results
            print(col_name)
            input_df[col_name] = series
    timer.time("done fitting to df")

    input_df.to_csv(output_file, float_format='%.6f', index=False)
    timer.time("done output")
