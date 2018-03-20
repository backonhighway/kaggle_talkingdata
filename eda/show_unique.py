import pandas as pd
import csv_loader
import seaborn as sns
import matplotlib.pyplot as plt

file_name = "../input/train.csv"
dtypes = csv_loader.get_dtypes()

df = pd.read_csv(file_name, dtype=dtypes, nrows=1000*1000*1)

df["ip_app_unique"] = df.groupby("ip")["app"].transform("nunique")
df["ip_channel_unique"] = df.groupby("ip")["channel"].transform("nunique")

wow = df.groupby("ip_app_unique")["is_attributed"].mean().reset_index()
print(wow.head())
wow.plot(kind="scatter", x="ip_app_unique", y="is_attributed")
plt.show()

wow = df.groupby("ip_channel_unique")["is_attributed"].mean().reset_index()
print(wow.head())

wow.plot(kind="scatter", x="ip_channel_unique", y="is_attributed")
plt.show()