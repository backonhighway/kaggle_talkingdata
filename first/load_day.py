import pandas as pd
import gc

file_name = "../input/train.csv"
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'click_time'    : 'object',
        'attributed_time'    : 'object',
        'is_attributed' : 'uint8',
        # 'click_id'      : 'uint32'
}
use_cols = ['ip','app','device', 'os', 'channel', 'click_time', 'is_attributed']

df = pd.read_csv(file_name, dtype=dtypes, usecols=use_cols)
df["click_time"] = pd.to_datetime(df["click_time"])
print("done loading...")

def output_by_day(df, day):
    temp_df = df[df["click_time"].dt.day == day]
    print("output to csv...")
    temp_df.to_csv('../input/train_day3.csv', float_format='%.6f', index=False)
    del temp_df
    gc.collect()


output_by_day(df, 8)
