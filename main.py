from utils.config import *
from utils.preprocessing import DataPrep
from utils.traxNet import newsModel 
import sys

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

    def predict_all_probs(text, model, vocab, max_len=32):
        # تبدیل متن به توکن
        tokens = preprocess.text_to_tensor(text, vocab)
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]

        tokens_array = np.array([tokens], dtype=np.int32)
        log_probs = model(tokens_array)[0]  # خروجی LogSoftmax

        probs = np.exp(log_probs)  # تبدیل log-probabilities به probabilities

        label_to_category = {v: k for k, v in Persian_categories.items()}

        print("\nاحتمال دسته‌بندی‌ها:")
        sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        for idx, prob in sorted_probs:
            category = label_to_category.get(idx, f"Unknown_{idx}")
            print(f"{category}: {prob:.4f}")

        pred_label = int(np.argmax(probs))
        return label_to_category[pred_label]

    while True:
        test_text = input(" جمله‌ی تستی (یا 0 برای خروج): ")
        if test_text.strip() == "0":
            break
        predicted_category = predict_all_probs(test_text, model, vocab)
        print("عنوان خبری پیش‌بینی‌شده:", predicted_category)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        trainer()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict()
    else:
        print("از 'train' یا 'predict' به عنوان ورودی استفاده کن.")
