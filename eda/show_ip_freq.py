import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
TRAIN_DATA = os.path.join(INPUT_DIR, "train.csv")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer

dtypes = csv_loader.get_dtypes()
df = pd.read_csv(TRAIN_DATA, dtype=dtypes, nrows=1000*1000*3)

#df["ip_count"] = df.groupby("ip")["channel"].transform("count")
#wow = df.groupby("ip_count")["is_attributed"].mean().reset_index()
#print(wow.head())
#wow.plot(kind="scatter", x="ip_count", y="is_attributed")

grouped = df.groupby("ip")["channel"].count().reset_index().sort_values(by="channel")
print(grouped.head(10))
count_group = grouped.groupby("channel").count().reset_index()
count_group.columns=["ip_count", "freq"]
print(count_group)
sns.distplot(count_group["ip_count"])
#count_group.plot(kind="scatter", x="channel", y="ip")
plt.show()

grouped = df.groupby(["ip","device","os"])["channel"].count().reset_index().sort_values(by="channel")
print(grouped.head(10))
count_group = grouped.groupby("channel")["ip"].count().reset_index()
count_group.columns=["idoa_count", "freq"]
print(count_group)
sns.distplot(count_group["idoa_count"])
#count_group.plot(kind="scatter", x="channel", y="ip")
plt.show()