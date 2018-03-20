import pandas as pd
import csv_loader
import seaborn as sns

file_name = "../input/train.csv"
dtypes = csv_loader.get_dtypes()

df = pd.read_csv(file_name, dtype=dtypes, nrows=1000*1000*10)
df['day'] = df.click_time.str[8:10].astype(int)
df['hour'] = df.click_time.str[11:13].astype(int)
# df["click_time"] = pd.to_datetime(df["click_time"])

df["ip_count"] = df.groupby(["ip"])["channel"].transform("count")
df["hourly_click_count"] = df.groupby(["ip", "day", "hour"])["channel"].transform('count')

print(df.head())

sns.distplot(df["hourly_click_count"])
sns.plt.show()
