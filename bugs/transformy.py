import pandas as pd
import csv_loader
import feature_engineerer

dtypes = csv_loader.get_dtypes()
train = pd.read_csv('../input/train_first_10k.csv', dtype=dtypes)

feature_engineerer.do_feature_engineering(train)

print(train.head())
train.to_csv("../output/checking.csv", index=False)