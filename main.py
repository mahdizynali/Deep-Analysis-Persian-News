from utils.config import *
from utils.torchNet import newsModel 
from utils.dataGen import NewsDataset
from utils.preprocessing import DataPrep

from sklearn.metrics import accuracy_score

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy


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
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = NewsDataset(X_train, y_train, vocab)
    test_dataset = NewsDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = newsModel(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCH + 1):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCH}", leave=False)
        for inputs, labels in progress:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_labels) if len(all_labels) == 0 else accuracy_score(all_labels, all_preds)

        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)

        print(f"[Epoch {epoch}] 🟩 Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | 🟦 Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "model/news_model_epoch{epoch}.pth")
    return model, test_loader



if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    # elif len(sys.argv) > 1 and sys.argv[1] == "predict":
    #     predict()
    else:
        print("از 'train' یا 'predict' به عنوان ورودی استفاده کن.")
