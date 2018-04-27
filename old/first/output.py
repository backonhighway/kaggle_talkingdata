import pandas as pd
import feature_engineerer

train = pd.read_csv('../input/train_first_1000k.csv')

feature_engineerer.do_feature_engineering(train)
print(train.describe())
print(train["ip"].nunique())
exit(0)

train.to_csv("../output/engineered_first_1000k.csv", float_format='%.6f', index=False)
