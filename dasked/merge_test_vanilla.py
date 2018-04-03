import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from dask import dataframe as dd
import pytz
from talkingdata.common import csv_loader, pocket_logger

TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
OLD_FILE = os.path.join(INPUT_DIR, "test_old.csv")

use_col = ["click_id", "ip", "app", "device", "os", "channel", "click_time"]
merge_col = ["ip", "app", "device", "os", "channel", "click_time"]
dtypes = csv_loader.get_dtypes()
test_df = dd.read_csv(TEST_FILE, dtype=dtypes, usecols=use_col).compute()
test_old_df = dd.read_csv(OLD_FILE, dtype=dtypes, usecols=merge_col).compute()
print(test_df.info())
print(test_old_df.info())


def get_time(df):
    df['day'] = df.click_time.str[8:10].astype(int)
    df['hour'] = df.click_time.str[11:13].astype(int)
    df['time_min'] = df.click_time.str[14:16].astype(int)
    df['time_sec'] = df.click_time.str[17:20].astype(int)


get_time(test_df)
get_time(test_old_df)
merge_col = ["ip", "app", "device", "os", "channel", "day", "hour", "time_min", "time_sec"]
test_df["rank"] = test_df.groupby(merge_col).cumcount().astype("int")
test_old_df["rank"] = test_old_df.groupby(merge_col).cumcount().astype("int")
merge_col.append("rank")

test_df = pd.merge(test_old_df, test_df, on=merge_col, how="left")
print(test_df.info())

submitting = test_df[test_df["click_id"].notnull()]
print(submitting.info())

cst = pytz.timezone('Asia/Shanghai')
test_df['click_time'] = pd.to_datetime(test_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
test_df = test_df.drop_duplicates(subset=['click_id'])
test_df = test_df[use_col]
print(test_df.info())

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_test_vanilla.csv")
test_df.to_csv(OUTPUT_FILE,  float_format='%.6f', index=False)