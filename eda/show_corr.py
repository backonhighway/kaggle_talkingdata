import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TEAM_DIR = os.path.join(OUTPUT_DIR, "team")
POCKET = os.path.join(TEAM_DIR, "pocket_prediction.csv")
MAMAS = os.path.join(TEAM_DIR, "submission_22_day3_pred.csv")
DANIJEL = os.path.join(TEAM_DIR, "29_LGB.csv")

import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer
from concurrent import futures

pocket_df = dd.read_csv(POCKET, header=None, usecols=[1]).compute()
mamas_df = dd.read_csv(MAMAS, header=None).compute()
danijel_df = dd.read_csv(DANIJEL, header=None).compute()
pocket_df.columns=["pred"]
mamas_df.columns=["pred"]
danijel_df.columns=["pred"]
print(pocket_df.describe())
print(mamas_df.describe())
print(danijel_df.describe())


def show_corr(df1, df2, name):
    corr_val = df1["pred"].corr(df2["pred"])
    return name, corr_val


def show_corr_spear(df1, df2, name):
    corr_val = df1["pred"].corr(df2["pred"], method="spearman")
    return name, corr_val


timer = pocket_timer.GoldenTimer()
with futures.ThreadPoolExecutor(max_workers=16) as executor:
    future_list = list()
    future_list.append(executor.submit(show_corr, pocket_df, mamas_df, "pm_corr"))
    future_list.append(executor.submit(show_corr, pocket_df, danijel_df, "pd_corr"))
    future_list.append(executor.submit(show_corr, danijel_df, mamas_df, "dm_corr"))
    future_list.append(executor.submit(show_corr_spear, pocket_df, mamas_df, "pm_spear"))
    future_list.append(executor.submit(show_corr_spear, pocket_df, danijel_df, "pd_spear"))
    future_list.append(executor.submit(show_corr_spear, danijel_df, mamas_df, "dm_spear"))
timer.time("done executor")

for one_future in future_list:
    result_tuple = one_future.result()
    print(result_tuple[0])
    print(result_tuple[1])

