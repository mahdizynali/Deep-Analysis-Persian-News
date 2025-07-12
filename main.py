from utils.config import *
from utils.preprocessing import DataPrep
from utils.traxNet import newsModel 

preprocess = DataPrep(DATASET_PATH)

print("\nPreparing Data ...\n")
X_train, X_test, y_train, y_test = preprocess.data_process()

print("\nCreating Vocab ...\n")
vocab = preprocess.word_vocab(np.concatenate([X_train, X_test]))

# print(f"Vocabulary size: {len(vocab)}")
# print("Sample words:", list(vocab.items()))

# tensor = preprocess.text_to_tensor("چه خوش گفت فردوسی پاکزاد", vocab=vocab)
# print(tensor)

#=============================================================================

def input_stream(preprocessor, data, labels, vocab):
    while True:
        for inputs, targets, weights in preprocessor.Generator(data, labels, vocab, loop=True, shuffle=True):
            yield (inputs.astype(np.int32), targets.astype(np.int32), weights.astype(np.float32))

def trainer():
    model = newsModel(len(vocab))
    train_stream = input_stream(preprocess, X_train, y_train, vocab)
    eval_stream = input_stream(preprocess, X_test, y_test, vocab)


    train_task = training.TrainTask(
        labeled_data=train_stream,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(LEARNING_RATE),
        n_steps_per_checkpoint=100
    )

    eval_task = training.EvalTask(
        labeled_data=eval_stream,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=10
    )

    training_loop = training.Loop(
        model=model,
        tasks=train_task,
        eval_tasks=[eval_task],
        output_dir='model_output'
    )

    print("Starting training...")
    training_loop.run(n_steps=ITRATION)
#===================================================================================

def predict():
    model = newsModel(len(vocab))
    model.init_from_file('model_output/model.pkl.gz', weights_only=True)

    def predict(text, model, vocab, max_len=32):
        tokens = preprocess.text_to_tensor(text, vocab)
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        tokens_array = np.array([tokens], dtype=np.int32)
        pred_probs = model(tokens_array)

        pred_label = np.argmax(pred_probs, axis=1).item()

        label_to_category = {v: k for k, v in Persian_categories.items()}
        return label_to_category[pred_label]


    while True:
        test_text = input(" جمله‌ی تستی (یا 0 برای خروج): ")
        if test_text.strip() == "0":
            break
        predicted_category = predict(test_text, model, vocab)
        print("عنوان خبری:", predicted_category)

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    {"train": trainer(), "predict": predict()}.get(cmd, lambda: print("❗ از 'train' یا 'predict' استفاده کنید."))()