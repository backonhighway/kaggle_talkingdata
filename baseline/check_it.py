import pandas as pd


train = pd.read_csv('../input/test_hour4.csv', nrows=1000)

print(train.head())