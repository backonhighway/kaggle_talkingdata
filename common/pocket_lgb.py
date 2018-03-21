import lightgbm as lgb
import pandas as pd
from . import pocket_logger

class GoldenLgb:
    def __init__(self):
        self.train_param = {
            'learning_rate': 0.02,
            'num_leaves': 31,
            'boosting': 'gbdt',
            'application': 'binary',
            'metric': 'auc',
            'feature_fraction': .7,
            'scale_pos_weight': 99,
            'seed': 99,
            'verbose': 0,
        }
        self.target_col_name = "is_attributed"
        self.category_col = [
            "app", "device", "os", "channel",
            "hour"
            #"ip_1", "ip_2", "ip_3", "ip_4", "ip_5", "ip_6",
        ]
        self.drop_cols = ["is_attributed"]

    def do_train(self, train_data, test_data):
        tcn = self.target_col_name
        y_train = train_data[tcn]
        y_test = test_data[tcn]
        x_train = train_data.drop(self.drop_cols, axis=1)
        x_test = test_data.drop(self.drop_cols, axis=1)

        return self.do_train_sk(x_train, x_test, y_train, y_test)

    def do_train_sk(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=lgb_eval,
                          verbose_eval=100,
                          num_boost_round=1000,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    @staticmethod
    def show_feature_importance(model):
        fi = pd.DataFrame({"name": model.feature_name(), "importance": model.feature_importance()})
        fi = fi.sort_values(by="importance", ascending=False)
        print(fi)
        logger = pocket_logger.get_my_logger()
        logger.info(fi)

