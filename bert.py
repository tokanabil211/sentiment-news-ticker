import pandas as pd
import psycopg2
from transformers import BertTokenizer, BertModel, get_scheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime
import json
import time
import os
import matplotlib.pyplot as plt

# Database config
db_config = {
    "host": "localhost",
    "port": "5432",
    "database": "financial_streaming",
    "user": "postgres",
    "password": "123456@Toka"
}

# Load data
def load_data():
    query = '''
        SELECT f.content, v.movement
        FROM fact_sentiment f
        JOIN training_data_by_sentiment_clean v ON f.sentiment_id = v.sentiment_id
        WHERE v.movement IN ('up', 'down')
    '''
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Dataset
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# Model
class BertClassifier(nn.Module):
    def __init__(self, bert_model):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(bert_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.sigmoid(self.fc(self.dropout(pooled)))

# Training function
def train_and_evaluate(epoch_count=17):
    df = load_data()
    X = df['content'].astype(str)
    y = (df['movement'] == 'up').astype(int)

    print(f"🔎 Class distribution: up={sum(y)}, down={len(y) - sum(y)}")

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X.to_frame(), y)
    X = X_res['content']
    y = y_res

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_set = BertDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    test_set = BertDataset(X_test.tolist(), y_test.tolist(), tokenizer)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16)

    model = BertClassifier(BertModel.from_pretrained('bert-base-uncased'))
    model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epoch_count)

    training_log = []

    for epoch in range(epoch_count):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(output, batch['label'].unsqueeze(1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch['input_ids'], batch['attention_mask']).round().squeeze()
                all_preds.extend(preds.tolist())
                all_labels.extend(batch['label'].tolist())

        acc = accuracy_score(all_labels, all_preds)
        training_log.append({"epoch": epoch + 1, "loss": total_loss, "accuracy": acc})
        print(f"📊 Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"bert_classifier_{timestamp}.pt"
    torch.save(model, model_path)
    tokenizer.save_pretrained(f"./bert_tokenizer_{timestamp}")
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Tokenizer saved to ./bert_tokenizer_{timestamp}")

    # Save log
    with open("bert_training_log.json", "w") as f:
        json.dump(training_log, f)

    print("\n📈 Final Report:")
    print(classification_report(all_labels, all_preds, target_names=["down", "up"]))

    # Plot
    log_df = pd.DataFrame(training_log)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(log_df["epoch"], log_df["loss"], marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(log_df["epoch"], log_df["accuracy"], color='green', marker='o')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("bert_training_plot.png")
    plt.show()

# Run loop
if __name__ == "__main__":
    print("🔁 Starting scheduled BERT training loop...")
    while True:
        print(f"⏰ {datetime.now()} - Training started")
        train_and_evaluate()
        print("⏳ Waiting 30 minutes...\n")
        time.sleep(1800)
