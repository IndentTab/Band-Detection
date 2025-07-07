import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# === Config ===
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 20
SNR_LIST = [-16, -14, -12, -10, -8, -6, -4, -2]  # For Case 3

# Adjust this path to where your synthetic data is saved
DATA_DIR = Path(r"C:\Users\XYZ\Collaborative learning\PartialObservation\RefinedNewData\SNRs\10m_Alpha5\250626_17_45")

# === Dataset ===
class SyntheticPSDCaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list):
        data = np.stack([np.array(x) for x in data_list])  # (N, 10, 1, 64, 20)
        data = data.squeeze(2)                             # (N, 10, 64, 20)
        N, SU, H, W = data.shape
        data = data.reshape(N * SU, H, W)                  # (N * 10, 64, 20)

        labels = np.repeat(np.array(label_list), repeats=10, axis=0)  # (N * 10, 20)

        self.X = torch.tensor(data, dtype=torch.float32)
        self.Y = torch.tensor(labels, dtype=torch.float32)

        self.X = (self.X - self.X.mean()) / self.X.std()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# === Model ===
class LSTMDetector(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 20, 64)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# === Evaluate PD at PFA = 5% ===
def evaluate_pd(model, test_loader, pfa_threshold=0.05):
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            y_true.append(yb.cpu())
            y_scores.append(preds.cpu())
    y_true = torch.cat(y_true).numpy().ravel()
    y_scores = torch.cat(y_scores).numpy().ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    try:
        pd_at_pfa = tpr[np.where(fpr >= pfa_threshold)[0][0]]
    except IndexError:
        pd_at_pfa = 0.0
    return pd_at_pfa

# === Main Loop ===
pd_results = []

for snr in SNR_LIST:
    file_path = DATA_DIR / f"Data_SNR{snr}vol20.pth"
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        continue

    print(f"\n📦 Running Case 3 @ SNR={snr}dB")
    db = torch.load(file_path, map_location=DEVICE)

    train_data = db['training data list']
    train_labels = db['training label list']
    test_data = db['testing data list']
    test_labels = db['testing label list']

    train_loader = DataLoader(SyntheticPSDCaseDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SyntheticPSDCaseDataset(test_data, test_labels), batch_size=BATCH_SIZE)

    model = LSTMDetector().to(DEVICE)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")

    # Save the model at SNR=-8dB for ROC comparison in later notebooks
    if snr == -8:
        torch.save(model.state_dict(), "lstm_case3_snr-8_model.pt")

    pd = evaluate_pd(model, test_loader, pfa_threshold=0.05)
    pd_results.append((snr, pd))

# === PD vs SNR Plot ===
snr_vals = [pt[0] for pt in pd_results]
pd_vals = [pt[1] for pt in pd_results]

plt.figure(figsize=(8, 5))
plt.plot(snr_vals, pd_vals, marker='o', label="LSTM (Case 3)", color='green')
plt.title("Case 3: PD vs SNR @ PFA=5%")
plt.xlabel("SNR (dB)")
plt.ylabel("Probability of Detection (PD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
