import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")

BASE_DATA7 = os.path.join(OUTPUT_DIR, "base_train_day7.csv")
BASE_DATA8 = os.path.join(OUTPUT_DIR, "base_train_day8.csv")
BASE_DATA9 = os.path.join(OUTPUT_DIR, "base_train_day9.csv")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from talkingdata.fe import runtime_fe, column_selector
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
train7 = dd.read_csv(TRAIN_DATA7, dtype=dtypes).compute()
train8 = dd.read_csv(TRAIN_DATA8, dtype=dtypes).compute()
train9 = dd.read_csv(TRAIN_DATA9, dtype=dtypes).compute()
timer.time("load csv in ")

use_col = [
    "ip", 'app', 'device', 'os', 'channel', 'hour', 'is_attributed',
    'group_i_count', "group_i_hourly_count_share", #"group_i_hourly_count",
    "group_ido_count", 'group_idoa_count', #'group_ioac_count', 'group_idoac_count',
    #"group_i_next_click_time", "group_i_prev_click_time"
    'group_ido_prev_click_time', 'group_ido_next_click_time',
    'group_idoa_prev_click_time', 'group_idoa_next_click_time',
    #'group_idoac_prev_click_time', 'group_idoac_next_click_time',
    'group_i_nunique_os', 'group_i_nunique_app',
    'group_i_nunique_channel', #'group_i_nunique_device',
    'group_i_ct_sum', 'group_i_ct_std', 'group_ido_ct_sum', 'group_ido_ct_std',
]

train7 = train7[use_col]
train8 = train8[use_col]
train9 = train9[use_col]

timer.time("start output")
train7.to_csv(BASE_DATA7, index=False, float_format='%.6f')
train8.to_csv(BASE_DATA8, index=False, float_format='%.6f')
train9.to_csv(BASE_DATA9, index=False, float_format='%.6f')
timer.time("done output")