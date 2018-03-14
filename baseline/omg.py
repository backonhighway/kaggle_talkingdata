import pandas as pd
import numpy as np

file_name = "../input/test.csv"
df = pd.read_csv(file_name, usecols=["click_id"])

df["is_attributed"] = np.where(df["click_id"] < 3344125, 1, 0)
df.to_csv("../output/lb_split.csv", float_format='%.6f', index=False)