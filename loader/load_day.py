import pandas as pd
import gc
import csv_loader
import pytz

file_name = "../input/train.csv"
dtypes = csv_loader.get_dtypes()
use_cols = ['ip','app','device', 'os', 'channel', 'click_time', 'is_attributed']

reader = pd.read_csv(file_name, dtype=dtypes, usecols=use_cols, chunksize=1000*1000)
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


