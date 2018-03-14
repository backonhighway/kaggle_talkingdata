import pandas as pd
import gc

file_name = "../input/test.csv"
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
use_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
reader = pd.read_csv(file_name, dtype=dtypes, usecols=use_cols, chunksize=1000000)

temp_df_list = []
public_hour = [4]
for tmp_df in reader:
    print("next_chunk")
    tmp_df["hour"] = pd.to_datetime(tmp_df["click_time"]).dt.hour.astype('uint8')
    tmp = tmp_df[tmp_df["hour"].isin(public_hour)]
    tmp.drop("hour", axis=1, inplace=True)
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print("done merge.")
df.to_csv("../input/test_hour4.csv", index=False)
