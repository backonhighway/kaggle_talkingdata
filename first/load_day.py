import pandas as pd
import gc
import csv_loader

file_name = "../input/train.csv"
dtypes = csv_loader.get_dtypes()
use_cols = ['ip','app','device', 'os', 'channel', 'click_time', 'is_attributed']

reader = pd.read_csv(file_name, dtype=dtypes, usecols=use_cols, chunksize=10000000)
print("done loading...")

temp_df_list = []
for tmp_df in reader:
    print("next_chunk")
    tmp_df["day"] = pd.to_datetime(tmp_df["click_time"]).dt.day.astype('uint8')
    tmp = tmp_df[tmp_df["day"] == 6]
    tmp.drop("day", axis=1, inplace=True)
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print("done merge.")

df.to_csv("../input/train_day1.csv", index=False)


