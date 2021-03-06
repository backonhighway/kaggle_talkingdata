import pandas as pd

file_name = "../input/train_day4.csv"
reader = pd.read_csv(file_name, chunksize=100000)

temp_df_list = []

for read_count in range(10):
    print("next_chunk")
    tmp = reader.get_chunk()

    temp_df_list.append(tmp)

result_df = pd.concat(temp_df_list)
print(result_df.describe())

print("output to csv...")
result_df.to_csv('../input/train_day3_first_1000k.csv', float_format='%.6f', index=False)




