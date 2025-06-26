import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
import os
from collections import Counter
import matplotlib.pyplot as plt

class EmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'embedding': torch.tensor(item['embedding'], dtype=torch.float),
            'label': torch.tensor(item['label'], dtype=torch.float)
        }

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.classifier(x).squeeze()

def evaluate_model(model, dataloader, device, threshold=0.3):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            emb = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            logits = model(emb)
            prob = torch.sigmoid(logits)
            probs.extend(prob.tolist())
            preds.extend((prob > threshold).long().tolist())
            targets.extend(labels.long().tolist())

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, zero_division=0)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(targets, probs)
    except:
        roc_auc = 0.0

    print("Evaluation Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

def plot_f1_vs_threshold(model, dataloader, device, output_path="f1_vs_threshold.png"):
    thresholds = [i / 100 for i in range(10, 91, 5)]
    f1_scores = []

    model.eval()
    for threshold in thresholds:
        preds, targets = [], []
        with torch.no_grad():
            for batch in dataloader:
                emb = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                logits = model(emb)
                prob = torch.sigmoid(logits)
                preds.extend((prob > threshold).long().tolist())
                targets.extend(labels.long().tolist())
        f1 = f1_score(targets, preds, zero_division=0)
        f1_scores.append(f1)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Threshold")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"📈 Threshold vs F1 curve saved to {output_path}")
    plt.show()

def train_model(train_path, val_path, epochs=5, batch_size=32, lr=1e-4, save_path="mlp_model.pt", only_eval=False):
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(val_path, 'r') as f:
        val_data = json.load(f)

    train_labels = [item['label'] for item in train_data]
    label_counts = Counter(train_labels)
    num_pos = label_counts.get(1, 1)
    num_neg = label_counts.get(0, 1)
    pos_weight = num_neg / num_pos

    train_dataset = EmbeddingDataset(train_data)
    val_dataset = EmbeddingDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier().to(device)

    if only_eval:
        model.load_state_dict(torch.load(save_path, map_location=device))
        evaluate_model(model, val_loader, device, threshold=0.3)
        plot_f1_vs_threshold(model, val_loader, device)
        return model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float).to(device))
    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            emb = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            logits = model(emb)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        print(f"\nEpoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f}")
        evaluate_model(model, val_loader, device, threshold=0.3)

        preds, targets = [], []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                emb = batch['embedding'].to(device)
                labels = batch['label'].to(device)
                logits = model(emb)
                prob = torch.sigmoid(logits)
                preds.extend((prob > 0.3).long().tolist())
                targets.extend(labels.long().tolist())
        f1 = f1_score(targets, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"🔥 Model saved to {save_path} (best F1: {best_f1:.4f})")

    if not os.path.exists(save_path):
        torch.save(model.state_dict(), save_path)
        print(f"✅ Final model saved to {save_path}")

    plot_f1_vs_threshold(model, val_loader, device)

    return model

def load_model(model_path="mlp_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
