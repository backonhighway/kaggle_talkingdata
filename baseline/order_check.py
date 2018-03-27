import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TEST_DATA = os.path.join(INPUT_DIR, "test.csv")

import pandas as pd
import numpy as np
from talkingdata.common import csv_loader

dtypes = csv_loader.get_featured_dtypes()
test = pd.read_csv(TEST_DATA, dtype=dtypes)
submission = pd.DataFrame({"click_id": test["click_id"]})
test["order_index"] = test.index
test["is_out_of_order"] = np.where(test["order_index"] < test["click_id"], 1, 0)
#print(test.head(30))
submission["is_attributed"] = test["is_out_of_order"]
print(submission.describe())
print("done prediction")

submission.to_csv("../output/order_check_submission.csv", index=False)