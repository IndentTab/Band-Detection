import h5py
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

file_path = r" insert path "
hf = h5py.File(file_path, 'r')
total_samples = hf['X'].shape[0]
subset_size = int(0.05 * total_samples)
indices = sorted(random.sample(range(total_samples), subset_size))  # sorted helps with faster reads

# ====== Read 5% only ======
X_small = hf['X'][indices]  # (subset_size, 1024, 2)
Y_small = hf['Y'][indices]  # (subset_size, 24)

print(f"Total samples: {total_samples}, Subset size: {subset_size}")

# ====== Custom Dataset Class ======
class RadioMLDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 1024, 2)
        self.Y = torch.tensor(np.argmax(Y, axis=1), dtype=torch.long)  # class index

        # Normalize IQ samples
        self.X = (self.X - self.X.mean()) / self.X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ====== Split Dataset ======
dataset = RadioMLDataset(X_small, Y_small)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

class LSTMModulationClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, num_classes=24):  
        super().__init__()  # FIXED
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
model = LSTMModulationClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ====== Training ======
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)  # weighted by batch size

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1:2d} | Avg Loss: {avg_loss:.4f}")

# ====== Evaluation ======
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
        all_labels.extend(yb.numpy())

print("\n=== Classification Report (5% sample) ===")
print(classification_report(all_labels, all_preds, digits=4))
