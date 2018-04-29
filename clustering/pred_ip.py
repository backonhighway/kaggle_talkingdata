import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
OUTPUT_CLUSTER = os.path.join(OUTPUT_DIR, "ip_cluster_day7.csv")

import pandas as pd
import numpy as np
import gc
from dask import dataframe as dd
from talkingdata.common import csv_loader, holdout_validator2, pocket_lgb, pocket_timer, pocket_logger
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()

train = pd.read_csv(OUTPUT_CLUSTER)
timer.time("load csv.")

predict_col = [
    "days", "count", "nunique_app", "nunique_device", "nunique_os",
    "prev_click_time_std"
]
train_y = train["is_attributed_mean"]
train_x = train[predict_col]
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

timer.time("prepare train in ")
lgb = pocket_lgb.get_ip_lgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)


