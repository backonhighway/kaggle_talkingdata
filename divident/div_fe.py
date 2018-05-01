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


def get_features(df, start_hour, window_size):
    df = get_time(df, start_hour, window_size)

    with futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_list = list()
        future_list.extend(submit_tasks(df, executor))
    timer.time("done executor")

    for one_future in future_list:
        list_of_tuples = one_future.result()
        for results in list_of_tuples:
            col_name, series = results
            df[col_name] = series
    timer.time("done fitting to df")

    return df


def get_time(df, start_hour, window_size):
    df["start_time"] = df["start_time"] + pd.DateOffset(hours=start_hour)
    df["end_time"] = df["start_time"] + pd.DateOffset(hours=window_size)
    df["time_till_start"] = df["click_time"] - df["start_time"]
    df["time_till_start"] = df["time_till_start"].dt.total_seconds()
    df["time_till_end"] = df["end_time"] - df["click_time"]
    df["time_till_end"] = df["time_till_end"].dt.total_seconds()
    return df


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
        #future_list.append(executor.submit(get_interval_click_time, df, name, grouping))

    n_unique_list = {
        "group_i": ["ip"]
    }
    for name, grouping in n_unique_list.items():
        # future_list.append(executor.submit(get_count_share, df, name, grouping))
        future_list.append(executor.submit(get_counts, df, name, grouping))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "os"))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "app"))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "channel"))
        future_list.append(executor.submit(get_nunique, df, name, grouping, "device"))
        #future_list.append(executor.submit(get_top_n_counts, df, name, grouping, 1))
        #future_list.append(executor.submit(get_top_n_counts, df, name, grouping, 2))
        #future_list.append(executor.submit(get_click_interval_and_stats, df, name, grouping))

    stats_list = {
        "group_idoa": ["ip", "app", "os", "device"],
        "group_ido": ["ip", "device", "os"],
    }
    #for name, grouping in stats_list.items():
        #future_list.append(executor.submit(get_click_interval_and_stats, df, name, grouping))

    return future_list


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
    # TODO should I use it?
    # ret_list.extend(get_rolling_time_diff(df, name, grouping, ret_list[0]))
    return ret_list


def get_interval_click_time0(df: pd.DataFrame, name: str, grouping:list):
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


# ------------------------------

def clip_it(df):
    prev_col_list = [
        'group_ido_prev_click_time', 'group_idoa_prev_click_time',
        'group_idoac_prev_click_time',
    ]
    for ct_col in prev_col_list:
        df[ct_col] = df[ct_col].clip(upper=df["time_till_start"])
        df[ct_col].fillna(df["time_till_start"])

    next_col_list = [
        'group_ido_next_click_time', 'group_idoa_next_click_time',
        'group_idoac_next_click_time',
    ]
    for ct_col in next_col_list:
        df[ct_col] = df[ct_col].clip(upper=df["time_till_end"])
        df[ct_col].fillna(df["time_till_end"])
    return df

