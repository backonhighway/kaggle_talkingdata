import pandas as pd
import feature_engineerer
import csv_loader

predictors = ['app','device','os', 'channel', 'hour', 'hourly_click_count']
categorical = ['app','device','os', 'channel', 'hour']
dtypes = csv_loader.get_dtypes()

test = pd.read_csv('../input/test.csv', dtype=dtypes, nrows=10000)
print(test.describe())
feature_engineerer.do_feature_engineering(test)
print(test.describe())

wtf = test[predictors]
print(wtf.describe())
