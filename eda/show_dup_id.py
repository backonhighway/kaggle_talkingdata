import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
TEST_DATA = os.path.join(INPUT_DIR, "test.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
use_col = feature_engineerer.get_necessary_col()
dtypes = csv_loader.get_dtypes()
test = dd.read_csv(TEST_DATA, dtype=dtypes).compute()
#print(train.info())
timer.time("load csv in ")

print(test.info())
test = test.drop_duplicates(subset=["click_id"])
print(test.info())