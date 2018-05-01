import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TEAM_DIR = os.path.join(OUTPUT_DIR, "team")
POCKET = os.path.join(TEAM_DIR, "pocket_prediction.csv")
DIVIDENT = os.path.join(TEAM_DIR, "sliding_window.csv")
MAMAS = os.path.join(TEAM_DIR, "submission_22_day3_pred.csv")
DANIJEL = os.path.join(TEAM_DIR, "29_LGB.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer
from concurrent import futures


timer = pocket_timer.GoldenTimer()
pocket_df = dd.read_csv(POCKET, header=None).compute()
divident_df = dd.read_csv(POCKET, header=None).compute()
pocket_df.columns=["pred"]
divident_df.columns=["pred"]
timer.time("done csv load")


def show_corr(df1, df2, name):
    corr_val = df1["pred"].corr(df2["pred"])
    return name, corr_val


def show_corr_spear(df1, df2, name):
    corr_val = df1["pred"].corr(df2["pred"], method="spearman")
    return name, corr_val


with futures.ThreadPoolExecutor(max_workers=16) as executor:
    future_list = list()
    future_list.append(executor.submit(show_corr, pocket_df, divident_df, "pd_corr"))
timer.time("done executor")

for one_future in future_list:
    result_tuple = one_future.result()
    print(result_tuple[0])
    print(result_tuple[1])

