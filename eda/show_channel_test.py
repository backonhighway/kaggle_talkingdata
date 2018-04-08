import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(INPUT_DIR, "test_old.csv")
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "channel_eda_test.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader

print("started")
dtypes = csv_loader.get_dtypes()
df = dd.read_csv(TRAIN_DATA, dtype=dtypes).compute()
grouped = df.groupby("channel")["ip"].count()
output_df = grouped.reset_index()
output_df.to_csv(OUTPUT_DATA, index=False)
