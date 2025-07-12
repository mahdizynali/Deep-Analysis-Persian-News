import os
import re
import string
import numpy as np
import pandas as pd
import random as rnd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import trax
from trax import layers as tl
from trax.supervised import training

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DATASET_PATH = "dataset"
NUMBERS_PATH = "manipulate/numbers.txt"
STOPWORDS_PATH = "manipulate/stopwords.txt"
PUNCTUATIONS_PATH = "manipulate/punctuations.txt"
STEMMER_SUFFIX_PATH = "manipulate/stem_suffixes.txt"
STEMMER_PREFIX_PATH = "manipulate/stem_prefixes.txt"

English_categories = {
    "sport": 0,
    "politics": 1,
    "internatial": 2,
    "social": 3,
    "economical": 4,
    "cultural": 5,
    "science": 6,
    "events": 7
}

Persian_categories = {
    "ورزشی": 0,
    "سیاسی": 1,
    "بین الملل": 2,
    "اجتماعی": 3,
    "اقتصادی": 4,
    "فرهنگی": 5,
    "علمی": 6,
    "حوادث": 7
}

LEARNING_RATE = 0.001
BATCH_SIZE = 16
CLASSES = 8
ITRATION = 5000