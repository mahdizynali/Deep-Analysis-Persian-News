import os
import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    0: "ورزشی",
    1: "سیاسی",
    2: "بین الملل",
    3: "اجتماعی",
    4: "اقتصادی",
    5: "فرهنگی",
    5: "علمی",
    7: "حوادث"
}

LEARNING_RATE=0.01
CLASSES=7 
ITRATION=1