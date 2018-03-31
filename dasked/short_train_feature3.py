import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from talkingdata.dasked import short_feature_module

TRAIN_FILE = os.path.join(INPUT_DIR, "train_day9.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "short_train_day9.csv")

short_feature_module.make_file(TRAIN_FILE, OUTPUT_FILE)
