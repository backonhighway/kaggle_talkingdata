import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader


def basic(df: pd.DataFrame):
    #df['day'] = df.click_time.str[8:10].astype(int)
    df['hour'] = df.click_time.str[11:13].astype(int)
    df["click_time"] = dd.to_datetime(df["click_time"])
    #df["ip_count"] = df.groupby("ip")["channel"].transform('count')
    #df["app_count"] = df.groupby("app")["channel"].transform('count')
    #df["os_count"] = df.groupby("os")["channel"].transform('count')
    df["telling_ip"] = np.where(df["ip"] <= 126420, 1, 0)


def do_pandas(df: pd.DataFrame):
    df["idoa_is_last_try"] = df.groupby(["ip", "app", "device", "os"])["channel"].shift(-1)
    df["idoa_is_last_try"] = np.where(df["idoa_is_last_try"].isnull(), 1, 0)


def do_grouping(df: pd.DataFrame):
    group_list = {
        "group_i": ["ip"],
        "group_idoa": ["ip", "app", "os", "device"],
        "group_ioac": ["ip", "app", "os", "channel"],
        "group_idoac": ["ip", "app", "os", "channel", "device"],
    }
    for name, grouping in group_list.items():
        get_counts(df, name, grouping)

    n_unique_list = {
        "group_i": ["ip"]
    }
    for name, grouping in n_unique_list.items():
        get_nunique(df, name, grouping, "os")
        get_nunique(df, name, grouping, "app")
        get_nunique(df, name, grouping, "channel")
        get_interval_click_time(df, name, grouping)

    user_group_list = {
        "group_ido": ["ip", "device", "os"],
    }
    for name, grouping in user_group_list.items():
        get_interval_click_time(df, name, grouping)
        get_short_stats(df, name, grouping)

    all_group_list = {
        "group_idoac": ["ip", "device", "os", "channel", "app"]
    }
    #for name, grouping in all_group_list.items():
    #    get_interval_click_time(df, name, grouping)


def get_interval_click_time(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    pct_col = name + "_prev_click_time"
    nct_col = name + "_next_click_time"
    df[pct_col] = grouper["click_time"].shift(1)
    df[nct_col] = grouper["click_time"].shift(-1)
    #df[pct_col] = df[pct_col].fillna()
    df[pct_col] = df["click_time"] - df[pct_col]
    df[pct_col] = df[pct_col].dt.total_seconds()
    df[nct_col] = df[nct_col] - df["click_time"]
    df[nct_col] = df[nct_col].dt.total_seconds()


def get_short_stats(df: pd.DataFrame, name: str, grouping:list):
    pct_col = name + "_prev_click_time"
    sum_col = name + "ct_sum"
    df[sum_col] = df.groupby(grouping)[pct_col].transform("sum")


def get_counts(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    cnt_col = name + "_count"
    df[cnt_col] = grouper["device"].transform("count")


def get_nunique(df: pd.DataFrame, name: str, grouping:list, nunique_col:str):
    grouper = df.groupby(grouping)
    uni_col = name + "_nunique"+ "_" + nunique_col
    df[uni_col] = grouper[nunique_col].transform("nunique")


def do_it_all(df: pd.DataFrame):
    basic(df)
    print("done basic features")
    do_grouping(df)
    print("done grouping features")


def make_file(input_file, output_file):
    dtypes = csv_loader.get_dtypes()
    input_df = dd.read_csv(input_file, dtype=dtypes).compute()
    do_it_all(input_df)

    input_df.to_csv(output_file, float_format='%.6f', index=False)
