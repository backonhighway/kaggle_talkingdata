import pandas as pd
import gc
import csv_loader

file_name = "../input/train_day4.csv"
dtypes = csv_loader.get_dtypes()
use_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
reader = pd.read_csv(file_name, dtype=dtypes, usecols=use_cols, chunksize=1000000)

temp_df_list = []

for tmp_df in reader:
    print("next_chunk")
    tmp_df["hour"] = pd.to_datetime(tmp_df["click_time"]).dt.hour.astype('uint8')
    tmp = tmp_df[(tmp_df["hour"] >= 4) & (tmp_df["hour"] <= 15)]
    tmp.drop("hour", axis=1, inplace=True)
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print("done merge.")
df.to_csv("../input/test_hour4.csv", index=False)
