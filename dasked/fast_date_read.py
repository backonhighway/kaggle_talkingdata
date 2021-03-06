import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TEST_FILE = os.path.join(INPUT_DIR, "train_day7.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "date_read_test.csv")

from concurrent import futures
import pandas as pd
import numpy as np
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer

timer = pocket_timer.GoldenTimer()
dtypes = csv_loader.get_dtypes()

# temp_df = pd.read_csv(TEST_FILE, dtype=dtypes, nrows=1000*1000*10)
# print(temp_df.info())
# temp_df.to_csv(OUTPUT_FILE, index=False)

###############################
# timer.time("start_parse")
# parsed_df = pd.read_csv(OUTPUT_FILE, dtype=dtypes, parse_dates=["click_time"], infer_datetime_format=True)
# print(parsed_df.info())
# timer.time("end_parse")
#
# input_df = dd.read_csv(OUTPUT_FILE, dtype=dtypes).compute()
# input_df["click_time"] = pd.to_datetime(input_df["click_time"])
# print(input_df.info())
# timer.time("done dd")
###############################

# input_df = dd.read_csv(OUTPUT_FILE, dtype=dtypes).repartition(npartitions=16)
# input_df["click_time"] = dd.to_datetime(input_df["click_time"])
# input_df = input_df.compute()
# print(input_df.info())
# timer.time("done dd")

# input_df = dd.read_csv(OUTPUT_FILE, dtype=dtypes, parse_dates=["click_time"]).compute()
# print(input_df.info())
# timer.time("done dd")

# pd_df = pd.read_csv(OUTPUT_FILE, dtype=dtypes)
# pd_df["click_time"] = pd.to_datetime(pd_df["click_time"])
# print(pd_df.info())
# timer.time("done pd")



def func(df):
    return pd.to_datetime(df["click_time"])

from sklearn import model_selection
from multiprocessing.dummy import Pool as ThreadPool

timer.time("start multi thread")
input_df = dd.read_csv(OUTPUT_FILE, dtype=dtypes).compute()

df_list = np.array_split(input_df, 16)

pool = ThreadPool(4)
results = pool.map(func, df_list)
series = pd.concat(results)
input_df["new_click_time"] = series
print(input_df.head())
print(input_df.info())
timer.time("done multi thread")

