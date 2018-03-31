import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import gc
from talkingdata.common import csv_loader
import pytz

INPUT_FILE = os.path.join(INPUT_DIR, "train.csv")
dtypes = csv_loader.get_dtypes()
use_cols = ['ip','app','device', 'os', 'channel', 'click_time', 'is_attributed']

reader = pd.read_csv(INPUT_FILE, dtype=dtypes, usecols=use_cols, chunksize=1000*1000*5)
print("done loading...")

temp_df_list = []
for tmp_df in reader:
    print("next_chunk")
    cst = pytz.timezone('Asia/Shanghai')
    tmp_df['click_time'] = pd.to_datetime(tmp_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
    tmp_df["day"] = tmp_df["click_time"].dt.day.astype('uint8')
    tmp = tmp_df[tmp_df["day"] == 7]
    tmp.drop("day", axis=1, inplace=True)
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print("done merge.")
print(len(df))
df.to_csv("../input/train_day7.csv", index=False)


