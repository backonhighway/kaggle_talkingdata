import pandas as pd
from sklearn import metrics
import feature_engineerer
import pocket_logger
import csv_loader


class HoldoutValidator:
    def __init__(self, model):
        self.logger = pocket_logger.get_my_logger()
        self.model = model
        dtypes = csv_loader.get_dtypes()
        self.holdout_df = pd.read_csv('../input/holding_test_hours.csv', dtype=dtypes)
        # do feature engineering
        feature_engineerer.do_feature_engineering(self.holdout_df)
        print("Initialized validator.")

    def validate(self):
        y_pred = self.model.predict(self.holdout_df)
        y_true = self.holdout_df["is_attributed"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)
