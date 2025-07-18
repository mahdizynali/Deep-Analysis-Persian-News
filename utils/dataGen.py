from utils.config import *
from .tokenizer import MaZe_tokenizer


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = MaZe_tokenizer()
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.do_tokenize(self.texts[idx])
        token_ids = [self.vocab.get(w, 0) for w in tokens][:self.max_len]
        padding = [0] * (self.max_len - len(token_ids))
        input_tensor = torch.tensor(token_ids + padding, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_tensor, label_tensor
