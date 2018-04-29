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


def get_counts(df: pd.DataFrame, name: str, grouping:list):
    grouper = df.groupby(grouping)
    cnt_col = name + "_count"
    series = grouper["channel"].transform("count")
    return [(cnt_col, series)]


def doit(df):
    group_list = {
        "group_i": ["ip"],
        "group_ido": ["ip", "device", "os"],
        "group_idoa": ["ip", "app", "os", "device"],
    }
    for name, grouping in group_list.items():
        get_counts(df , name, grouping)



pd.DateOffset(hours=16)