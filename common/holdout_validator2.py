import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")

import pandas as pd
from dask import dataframe as dd
from sklearn import metrics
from talkingdata.common import csv_loader, feature_engineerer, pocket_logger


class HoldoutValidator:
    def __init__(self, model, df, predict_col):
        self.logger = pocket_logger.get_my_logger()
        self.model = model
        self.holdout_df = df
        self.predict_col = predict_col
        print("Initialized validator.")

    def validate(self):
        self.holdout_df["pred"] = self.model.predict(self.holdout_df[self.predict_col])

        y_true = self.holdout_df["is_attributed"]
        y_pred = self.holdout_df["pred"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)

        public_df = self.holdout_df[self.holdout_df["hour"] == 12]
        y_true = public_df["is_attributed"]
        y_pred = public_df["pred"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)

        # not exact, but close enough...
        private_hour_list = [13, 17, 18, 21, 22]
        private_df = self.holdout_df[self.holdout_df["hour"].isin(private_hour_list)]
        y_true = private_df["is_attributed"]
        y_pred = private_df["pred"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)

        private_df2 = self.holdout_df[self.holdout_df["hour"] >= 13]
        y_true = private_df2["is_attributed"]
        y_pred = private_df2["pred"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)

    def validate_rmse(self):
        y_true = self.holdout_df["is_attributed"]
        y_pred = self.holdout_df["pred"]
        score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
        print(score)
        self.logger.info(score)

        eval_df = self.holdout_df[["ip", "hour"]].copy()
        eval_df["diff"] = abs(y_pred - y_true)
        grouped = eval_df.groupby("ip")["diff"].agg({"count", "mean"}).\
            reset_index().sort_values(by="mean", ascending=False)
        grouped = grouped[grouped["count"] > 1000]
        print(grouped.head(20))
        self.logger.info(grouped.head(30))

