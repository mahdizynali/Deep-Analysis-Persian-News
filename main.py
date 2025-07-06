from utils.config import *
from utils.preprocessing import DataPrep

preprocess = DataPrep(DATASET_PATH)

print("\nPreparing Data ...\n")
X_train, X_test, y_train, y_test = preprocess.data_process()

print("\nCreating Vocab ...\n")
vocab = preprocess.word_vocab(np.concatenate([X_train, X_test]))

# print(f"Vocabulary size: {len(vocab)}")
# print("Sample words:", list(vocab.items()))

tensor = preprocess.text_to_tensor("چه خوش گفت فردوسی پاکزاد", vocab=vocab)