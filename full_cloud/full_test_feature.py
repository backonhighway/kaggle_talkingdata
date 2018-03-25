import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from talkingdata.full_cloud import full_feature_module

TEST_FILE = os.path.join(INPUT_DIR, "test.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "full_test_featured.csv")

full_feature_module.make_file(TEST_FILE, OUTPUT_FILE)
