from .config import *

def newsModel(vocab_size, d_model=128):
    return tl.Serial(
        tl.Embedding(vocab_size, d_feature=d_model),
        tl.LSTM(d_model),
        tl.Mean(axis=1),
        tl.Dense(128),
        tl.Relu(),
        tl.Dense(64),
        tl.Relu(),
        tl.Dense(CLASSES),
        tl.LogSoftmax()
    )