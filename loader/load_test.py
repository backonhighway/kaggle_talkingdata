import pandas as pd
import gc
import csv_loader


def get_mask(df, start_datetime, end_datetime):
    return (df['click_time'] >= start_datetime) & (df['click_time'] <= end_datetime)


file_name = "../input/holdout_d4h4.csv"
dtypes = csv_loader.get_dtypes()
use_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
reader = pd.read_csv(file_name, dtype=dtypes, usecols=use_cols, chunksize=1000000)

temp_df_list = []

for tmp_df in reader:
    print("next_chunk")
    mask1 = get_mask(tmp_df, "2017-11-09 04:00:00", "2017-11-09 06:00:00")
    mask2 = get_mask(tmp_df, "2017-11-09 09:00:00", "2017-11-09 11:00:00")
    mask3 = get_mask(tmp_df, "2017-11-09 13:00:00", "2017-11-09 15:00:00")
    tmp = tmp_df[mask1 | mask2 | mask3]
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print("done merge.")
print(df.info())
df.to_csv("../input/holding_test_hours.csv", index=False)
