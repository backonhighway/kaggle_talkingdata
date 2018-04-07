import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TEST_FILE = os.path.join(INPUT_DIR, "train_day7.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "date_read_test.csv")

from concurrent import futures
import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer

timer = pocket_timer.GoldenTimer()
dtypes = csv_loader.get_dtypes()
input_df = dd.read_csv(TEST_FILE, dtype=dtypes).repartition(npartitions=64)
timer.time("done read")

def get_dd_time(df: dd.DataFrame):
    df["click_time"] = dd.to_datetime(df["click_time"], )
    timer.time("done click time")
    df["hour"] = df["click_time"].dt.hour
    #df["telling_ip"] = np.where(df["ip"] <= 126420, 1, 0)


