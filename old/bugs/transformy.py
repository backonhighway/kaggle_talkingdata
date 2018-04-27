import pandas as pd
import csv_loader
import feature_engineerer

dtypes = csv_loader.get_dtypes()
df = pd.read_csv('../input/test.csv', dtype=dtypes, nrows=100000)

feature_engineerer.do_feature_engineering(df)

print(df.head())
df.to_csv("../output/checking.csv", index=False)