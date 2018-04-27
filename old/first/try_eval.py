import pandas as pd
import numpy as np
from sklearn import metrics

train = pd.read_csv('../input/train_sample.csv')

train["predict"] = 0
train["predict_2"] = 1
train["predict_3"] = 0.01

print(train.head())
print("-"*40)

score = metrics.roc_auc_score(train["is_attributed"], train["predict"])
print(score)
score = metrics.roc_auc_score(train["is_attributed"], train["predict_2"])
print(score)
score = metrics.roc_auc_score(train["is_attributed"], train["predict_3"])
print(score)
