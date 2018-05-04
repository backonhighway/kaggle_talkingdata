import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "long_test.feather")
MAMAS_INDEX = os.path.join(INPUT_DIR, "last_test_idx.npy")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sub_long.csv")
OLD_INDEX = os.path.join(INPUT_DIR, "test_index.csv")
OUTPUT_MAMAS = os.path.join(OUTPUT_DIR, "sub_mamas_fukugen.csv")

import pandas as pd
import numpy as np
from sklearn import model_selection
import gc
from dask import dataframe as dd
from talkingdata.fe import column_selector, runtime_fe
from talkingdata.common import csv_loader, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

predict_col = column_selector.get_predict_col()
dtypes = csv_loader.get_featured_dtypes()

test_old_id = dd.read_csv(OLD_INDEX).compute()
print(test_old_id.head())

pred_df = dd.read_csv(OUTPUT_FILE).compute()
test = pd.merge(test_old_id, pred_df, on="click_id", how="left")
test = test.sort_values("old_click_id")
print(test.head())
test["is_attributed"] = test["is_attributed"].fillna(-1)

y_pred = np.array(test["is_attributed"])
mask = test["click_id"] >= 0
test = test[mask]
mamas_idx = np.load(MAMAS_INDEX)
submission = pd.DataFrame({"click_id": test["click_id"].astype("int")})
submission["is_attributed"] = y_pred[mamas_idx]
print(submission.describe())
submission = submission[submission["is_attributed"] == -1]
print(submission.describe())

exit(0)
submission.to_csv(OUTPUT_MAMAS, index=False)
timer.time("submission in ")

