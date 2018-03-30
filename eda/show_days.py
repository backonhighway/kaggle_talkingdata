import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA = os.path.join(OUTPUT_DIR, "train.csv")
HOLDOUT_DATA = os.path.join(OUTPUT_DIR, "full_train_day4_featured.csv")
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from talkingdata.common import csv_loader, feature_engineerer, holdout_validator, pocket_lgb, pocket_timer

dtypes = csv_loader.get_dtypes()
df = pd.read_csv(TRAIN_DATA, dtype=dtypes, nrows=1000*1000*1)

df["ip_count"] = df.groupby("ip")["channel"].transform("count")

wow = df.groupby("ip_count")["is_attributed"].mean().reset_index()
print(wow.head())

wow.plot(kind="scatter", x="ip_count", y="is_attributed")

plt.show()