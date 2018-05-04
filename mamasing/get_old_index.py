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
from talkingdata.common import csv_loader, pocket_logger, pocket_timer

TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
OLD_FILE = os.path.join(INPUT_DIR, "test_old.csv")

use_col = ["click_id", "ip", "app", "device", "os", "channel", "click_time"]
dtypes = csv_loader.get_dtypes()
timer = pocket_timer.GoldenTimer()

test_df = dd.read_csv(TEST_FILE, dtype=dtypes, usecols=use_col).compute()
test_old_df = dd.read_csv(OLD_FILE, dtype=dtypes).compute()

test_old_df["old_click_id"] = test_old_df["click_id"]
old_df_col = ["ip", "app", "device", "os", "channel", "click_time", "old_click_id"]
test_old_df = test_old_df[old_df_col]
print(test_df.info())
print(test_old_df.info())


timer.time("before ranking")
merge_col = ["ip", "app", "device", "os", "channel", "click_time"]
test_df["rank"] = test_df.groupby(merge_col).cumcount().astype("int")
test_old_df["rank"] = test_old_df.groupby(merge_col).cumcount().astype("int")

timer.time("before merge")
merge_col.append("rank")
test_df = pd.merge(test_old_df, test_df, on=merge_col, how="left")
test_df["click_id"] = test_df["click_id"].fillna(-1)
test_df["click_id"] = test_df["click_id"].astype("int")
timer.time("done merge")
print(test_df.info())

output_col = ["click_id", "old_click_id"]
test_df = test_df[output_col]
OUTPUT_FILE = os.path.join(INPUT_DIR, "test_index.csv")
test_df.to_csv(OUTPUT_FILE,  float_format='%.6f', index=False)