import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.embedding import EmbeddingClient
from utils.weaviate_utils import cargar_credenciales

class TripletDataset(Dataset):
    def __init__(self, triplets, embedder):
        self.data = []
        for triplet in triplets:
            q = embedder.generate_embedding(triplet["question"])
            p = embedder.generate_embedding(triplet["positive_link"])
            n = embedder.generate_embedding(triplet["negative_link"])
            if q and p and n:
                self.data.append((q, p, n))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tuple(torch.tensor(x, dtype=torch.float32) for x in self.data[idx])

class TripletNet(nn.Module):
    def __init__(self, embedding_dim=1536):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 512)

    def forward(self, x):
        return self.linear(x)

def train_triplet_model(triplet_path, model_path="data/rank_model.pt", epochs=10):
    config = cargar_credenciales()
    embedder = EmbeddingClient(api_key=config["OPENAI_API_KEY"], model="text-embedding-ada-002")

    with open(triplet_path, "r") as f:
        triplets = json.load(f)

    dataset = TripletDataset(triplets, embedder)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TripletNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in loader:
            out_a = model(anchor)
            out_p = model(positive)
            out_n = model(negative)
            loss = loss_fn(out_a, out_p, out_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_triplet_model("data/train_triplets.json")
