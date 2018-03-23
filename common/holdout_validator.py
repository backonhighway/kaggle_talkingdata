import os, sys
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.append(APP_ROOT)
INPUT_DIR = os.path.join(APP_ROOT, "input")
HOLDOUT_DATA = os.path.join(INPUT_DIR, "holding_test_hours.csv")

import pandas as pd
from sklearn import metrics
from talkingdata.common import csv_loader, feature_engineerer, pocket_logger

class HoldoutValidator:
    def __init__(self, model):
        self.logger = pocket_logger.get_my_logger()
        self.model = model
        dtypes = csv_loader.get_dtypes()
        self.holdout_df = pd.read_csv(HOLDOUT_DATA, dtype=dtypes)
        # do feature engineering
        feature_engineerer.do_feature_engineering(self.holdout_df)
        print("Initialized validator.")

    def validate(self):
        y_pred = self.model.predict(self.holdout_df)
        y_true = self.holdout_df["is_attributed"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)
