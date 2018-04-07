import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")

import pandas as pd
from dask import dataframe as dd
from sklearn import metrics
from talkingdata.common import csv_loader, feature_engineerer, pocket_logger


class HoldoutValidator:
    def __init__(self, model, df):
        self.logger = pocket_logger.get_my_logger()
        self.model = model
        self.holdout_df = df
        print("Initialized validator.")

    def validate(self):
        y_true = self.holdout_df["is_attributed"]
        y_pred = self.model.predict(self.holdout_df.drop("is_attributed", axis=1))
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)
