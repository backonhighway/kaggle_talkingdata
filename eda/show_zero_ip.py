import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
POCKET = os.path.join(OUTPUT_DIR, "005_val.csv")
LONG7 = os.path.join(OUTPUT_DIR, "long_train_day7.feather")
LONG8 = os.path.join(OUTPUT_DIR, "long_train_day8.feather")
LONG9 = os.path.join(OUTPUT_DIR, "long_train_day9.feather")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

pocket_df = dd.read_csv(POCKET, header=None).compute()
pocket_df.columns = ["pred"]
print(pocket_df.head())
pocket_array = np.array(pocket_df["pred"])

train7 = pd.read_feather(LONG7)
train8 = pd.read_feather(LONG8)
train9 = pd.read_feather(LONG9)
timer.time("load csv in ")

use_col = ["ip", "group_i_count", "is_attributed"]
train9 = train9[use_col]
train9["pred"] = pocket_array
print("-"*40)

train = train7.append(train8)
ip_mean = train.groupby("ip")["is_attributed"].mean().reset_index()
ip_mean.columns = ["ip", "mean"]
print(ip_mean.head())
mask_bad = ip_mean["mean"] <= 0.0
# bad_ip = ip_mean[mask_bad]
# print(bad_ip)
print("-"*40)

merged = pd.merge(train9, ip_mean, on="ip", how="left")
print(merged.head())
timer.time("done merge")


from sklearn import metrics

y_true = merged["is_attributed"]
y_pred = merged["pred"]
score = metrics.roc_auc_score(y_true, y_pred)
print(score)

mask = (merged["mean"] <= 0.0) & (merged["group_i_count"] >= 2000)
merged["pred"] = np.where(mask, 0, merged["pred"])

y_pred = merged["pred"]
score = metrics.roc_auc_score(y_true, y_pred)
print(score)
