import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TRAIN_DATA7 = os.path.join(OUTPUT_DIR, "short_train_day7.csv")
TRAIN_DATA8 = os.path.join(OUTPUT_DIR, "short_train_day8.csv")
TRAIN_DATA9 = os.path.join(OUTPUT_DIR, "short_train_day9.csv")
OUTPUT_CLUSTER = os.path.join(OUTPUT_DIR, "ip_cluster_day7.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ip_cluster_day7_res.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "test_plot.png")

import pandas as pd
import numpy as np
import gc
from sklearn import manifold
from dask import dataframe as dd
from talkingdata.common import csv_loader, pocket_timer, pocket_logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot


logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
ip_df = dd.read_csv(OUTPUT_CLUSTER).compute()
cluster_col = [
    "count", "nunique_app", "nunique_device", "nunique_os", #"prev_click_time_std",
]
print(ip_df.describe())

timer.time("start click scaling")

timer.time("do cluster")
model = manifold.TSNE(n_components=2, random_state=99)
result = model.fit_transform(ip_df[cluster_col])
timer.time("done cluster")

# I guess we can merge it with original df?
print(result)
# result_df = pd.DataFrame(result)
# result_df.columns=["X", "Y"]
# print(result_df.describe())
# result_df.plot(kind="scatter", x="X", y="Y")
# plot.savefig(OUTPUT_PNG)

result_df = pd.DataFrame(result)
result_df.columns=["X", "Y"]
ip_df["X"] = result_df["X"]
ip_df["Y"] = result_df["Y"]
print(ip_df.head())
ip_df.to_csv()

def plot_em(some_2d_array):
    xmin = some_2d_array[:,0].min()
    xmax = some_2d_array[:,0].max()
    ymin = some_2d_array[:,1].min()
    ymax = some_2d_array[:,1].max()
    plot.xlabel("component 0")
    plot.ylabel("component 1")
    plot.title("t-SNE visualization")
    plot.savefig("tsne.png")

#plot.savefig("ip_cluster.png")


exit(0)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit(ip_df)

ip_df['cluster'] = pd.Series(clusters.labels_, index=ip_df.index)