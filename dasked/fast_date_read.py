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
timer.time("start_parse")
parsed_df = pd.read_csv(TEST_FILE, dtype=dtypes, parse_dates="click_time", infer_datetime_format=True)
print(parsed_df["click_time"].info())
timer.time("end_parse")

input_df = dd.read_csv(TEST_FILE, dtype=dtypes).repartition(npartitions=32)
input_df["click_time"] = dd.to_datetime(input_df["click_time"])
input_df = input_df.compute()
print(input_df["click_time"].info())
timer.time("done dd")

pd_df = pd.read_csv(TEST_FILE, dtype=dtypes)
pd_df["click_time"] = pd.to_datetime(pd_df["click_time"])
print(pd_df["click_time"].info())
timer.time("done pd")



