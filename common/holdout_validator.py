import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
HOLDOUT_DATA = os.path.join(OUTPUT_DIR, "holdout_small_featured.csv")

import pandas as pd
from dask import dataframe as dd
from sklearn import metrics
from talkingdata.common import csv_loader, feature_engineerer, pocket_logger


class HoldoutValidator:
    def __init__(self, model, filename=None, num_rows=None):
        self.logger = pocket_logger.get_my_logger()
        self.model = model

        use_col = feature_engineerer.get_necessary_col()
        dtypes = csv_loader.get_featured_dtypes()
        local_filename = HOLDOUT_DATA
        if filename is None:
            filename = HOLDOUT_DATA
        if num_rows is None:
            self.holdout_df = dd.read_csv(filename, dtype=dtypes, usecols=use_col).compute()
        else:
            self.holdout_df = dd.read_csv(filename, dtype=dtypes, nrows=num_rows, usecols=use_col).compute()

        print(self.holdout_df.info())
        print("Initialized validator.")

    def validate(self):
        y_true = self.holdout_df["is_attributed"]
        y_pred = self.model.predict(self.holdout_df.drop("is_attributed", axis=1))
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)
