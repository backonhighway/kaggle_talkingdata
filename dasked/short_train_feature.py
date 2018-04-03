import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "talkingdata")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")

import pandas as pd
import numpy as np
from talkingdata.fe import dasked_fe

TRAIN_FILE = os.path.join(INPUT_DIR, "train_day7.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "short_train_day7.csv")

dasked_fe.make_file(TRAIN_FILE, OUTPUT_FILE)
