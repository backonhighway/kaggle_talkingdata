import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from talkingdata.common import csv_loader

TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
OLD_FILE = os.path.join(OUTPUT_DIR, "full_test_old_featured.csv")

dtypes = csv_loader.get_featured_dtypes()
test_df = pd.read_csv(TEST_FILE, dtype=dtypes)
test_old_df = pd.read_csv(OLD_FILE, dtype=dtypes)

test_df = pd.merge(test_df, test_old_df, on=["ip", "click_time", "app", "device", "os", "channel"], how="left")
test_df = test_df.drop("click_id_y", axis=1)
test_df.rename(columns={'click_id_x':'log(gdp)'}, inplace=True)
print(test_df.info())
print(test_df.head(30))

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_test.csv")
test_df.to_csv(OUTPUT_FILE,  float_format='%.6f', index=False)