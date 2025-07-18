import os
import re
import sys
import string
import numpy as np
import pandas as pd
import random as rnd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
CLASSES = 8
ITRATION = 5000
EPOCH = 15