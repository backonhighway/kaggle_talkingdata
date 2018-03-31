import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_logger

TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
OLD_FILE = os.path.join(INPUT_DIR, "test_old.csv")

use_col = ["click_id", "ip", "app", "device", "os", "channel", "click_time"]
dtypes = csv_loader.get_dtypes()
test_df = dd.read_csv(TEST_FILE, dtype=dtypes, usecols=use_col).compute()
use_col.remove("click_id")
test_old_df = dd.read_csv(OLD_FILE, dtype=dtypes, usecols=use_col).compute()
print(test_df.info())
print(test_old_df.info())

test_df = pd.merge(test_old_df, test_df, on=["ip", "click_time", "app", "device", "os", "channel"], how="left")
print(test_df.info())

logger = pocket_logger.get_my_logger()
logger.info(test_df.describe())
logger.info(test_df.head(10))
logger.info(test_df.tail(10))

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_test_vanilla.csv")
test_df.to_csv(OUTPUT_FILE,  float_format='%.6f', index=False)