import pandas as pd
import gc
import csv_loader


def get_mask(df, start_datetime, end_datetime):
    return (df['click_time'] >= start_datetime) & (df['click_time'] <= end_datetime)


file_name = "../input/test_old.csv"
dtypes = csv_loader.get_dtypes()
reader = pd.read_csv(file_name, dtype=dtypes, chunksize=1000*1000)

temp_df_list = []

for tmp_df in reader:
    print("next_chunk")
    mask1 = get_mask(tmp_df, "2017-11-10 04:00:00", "2017-11-10 04:02:00")
    tmp = tmp_df[mask1]
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print("done merge.")
print(df.info())
df.to_csv("../input/test_old_compare.csv", index=False)
