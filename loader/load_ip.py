import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(INPUT_DIR, "train.csv")
TRAIN_DATA7 = os.path.join(INPUT_DIR, "train_day7.csv")
TRAIN_DATA8 = os.path.join(INPUT_DIR, "train_day8.csv")
TRAIN_DATA9 = os.path.join(INPUT_DIR, "train_day9.csv")
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "show_ip_66831.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader

print("started")
dtypes = csv_loader.get_dtypes()
df = dd.read_csv(TRAIN_DATA, dtype=dtypes).compute()
print("loaded data")
df = df[df["ip"] == 66831]
df.to_csv(OUTPUT_DATA, float_format='%.3f', index=False)
