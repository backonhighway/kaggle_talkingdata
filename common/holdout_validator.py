import pandas as pd
from sklearn import metrics
import feature_engineerer
import pocket_logger


class HoldoutValidator:
    def __init__(self, model):
        self.logger = pocket_logger.get_my_logger()
        self.model = model
        self.holdout_df = pd.read_csv('../input/train_first_1000k.csv')
        # do feature engineering
        feature_engineerer.do_feature_engineering(self.holdout_df)
        print("Initialized validator.")

    def validate(self):
        y_pred = self.model.predict(self.holdout_df)
        y_true = self.holdout_df["is_attributed"]
        score = metrics.roc_auc_score(y_true, y_pred)
        print(score)
        self.logger.info(score)
