from .config import *

class newsModel():

    def __init__ (self, vocab_size) -> None :
        self.vcs = vocab_size
        self.Model()

    def Model(self, , d_model = 128):
        model_sequence = tl.Serial(
            tl.Embedding(self.vcs, d_feature=d_model),
            tl.LSTM(d_model),
            tl.Mean(axis=1),
            tl.Dense(128),
            tl.Relu(),
            tl.Dense(64),
            tl.Relu(),
            tl.Dense(CLASSES),
            tl.LogSoftmax()
        )
        return model_sequence