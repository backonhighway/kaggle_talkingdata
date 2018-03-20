import pandas as pd
import gc
import csv_loader

file_name = "../input/test.csv"
dtypes = csv_loader.get_dtypes()

reader = pd.read_csv(file_name, dtype=dtypes,  chunksize=1000*1000*10)
print("done loading...")

temp_df_list = []
for tmp_df in reader:
    print("next_chunk")
    tmp_df["index1"] = tmp_df.index
    tmp = tmp_df[tmp_df["index1"] != tmp_df["click_id"]]
    if tmp is not None:
        temp_df_list.append(tmp)

df = pd.concat(temp_df_list)
print(df.describe())
print("done merge.")

df.to_csv("../input/test_not_same.csv", index=False)


