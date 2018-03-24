import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
HOLDOUT_DATA = os.path.join(OUTPUT_DIR, "holdout_small_featured.csv")

import pandas as pd
from sklearn import metrics
from talkingdata.common import csv_loader, feature_engineerer, pocket_logger


class HoldoutValidator:
    def __init__(self, model):
        self.logger = pocket_logger.get_my_logger()
        self.model = model

        dtypes = csv_loader.get_featured_dtypes()
        self.holdout_df = pd.read_csv(HOLDOUT_DATA, dtype=dtypes)
        print(self.holdout_df.describe())

        # do feature engineering
        self.holdout_df = self.holdout_df[feature_engineerer.get_necessary_col()]
        print("Initialized validator.")

    def validate(self):
        y_pred = self.model.predict(self.holdout_df)
        y_true = self.holdout_df["is_attributed"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)
