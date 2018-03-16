import pandas as pd


train = pd.read_csv('../output/submission.csv', nrows=1000)

print(train.head())