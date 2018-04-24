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


def get_top_n_counts(df: pd.DataFrame, name: str, grouping:list, n: int):
    grouping = df.groupby(grouping)["device"]
    num_series = grouping.transform(lambda x: x.value_counts().nlargest(n).sum())
    count_series = grouping.transform("count")
    n_count_col = name + "_top" + str(n) + "_device_share"
    n_count_series = num_series / count_series
    n_count_series = n_count_series.multiply(100).round()
    return [(n_count_col, n_count_series)]


TEST_FILE = os.path.join(INPUT_DIR, "merged_test_vanilla.csv")

dtypes = csv_loader.get_dtypes()
input_df = dd.read_csv(TEST_FILE, dtype=dtypes).head(1000*1000*10)

reslist = get_top_n_counts(input_df, "aaa", ["ip"], 2)
print(reslist[0][0])
print(reslist[0][1])
