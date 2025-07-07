import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# === Environment Config ===
DEVICE = torch.device("cpu")
BATCH_SIZE = 64
EPOCHS = 20
SNR_LIST = [-4, -10, -16]  # For Case 1 and 2
DATA_DIR = Path(r"C:/Users/Srijita Saha/Collaborative learning/PartialObservation/RefinedNewData/SNRs/10m_Alpha5/250626_17_45")

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

# === LSTM Model ===
class LSTMDetector(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=2, num_classes=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # → (batch, 20, 64)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# === Evaluation Helper ===
def evaluate_model(model, loader):
    model.eval()
    y_true, y_scores = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            y_true.append(yb.cpu())
            y_scores.append(preds.cpu())
    return torch.cat(y_true).numpy(), torch.cat(y_scores).numpy()

# === Containers for Graph Data ===
losses_per_case = {"Case 1": {}, "Case 2": {}}
roc_per_case = {"Case 1": {}, "Case 2": {}}

# === Training Loop for Case 1 & 2 ===
for snr in SNR_LIST:
    file_path = DATA_DIR / f"Data_SNR{snr}vol20.pth"
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        continue

    db = torch.load(file_path, map_location=DEVICE)
    train_data = db['training data list']
    train_labels = db['training label list']
    test_data = db['testing data list']
    test_labels = db['testing label list']
    full_train_loader = DataLoader(SyntheticPSDCaseDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    small_train_loader = DataLoader(SyntheticPSDCaseDataset(train_data[:500], train_labels[:500]), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SyntheticPSDCaseDataset(test_data, test_labels), batch_size=BATCH_SIZE)

    for case, loader in zip(["Case 1", "Case 2"], [full_train_loader, small_train_loader]):
        print(f"\n Training {case} @ SNR={snr}dB")
        model = LSTMDetector().to(DEVICE)
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epoch_losses = []

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(loader.dataset)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")

        losses_per_case[case][snr] = epoch_losses

        # ROC Curve
        y_true, y_scores = evaluate_model(model, test_loader)
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_scores.ravel())
        roc_per_case[case][snr] = (fpr, tpr)

# === Plot: Convergence (Accuracy Proxy)
for case in ["Case 1", "Case 2"]:
    plt.figure(figsize=(7, 5))
    for snr in SNR_LIST:
        losses = losses_per_case[case][snr]
        accuracy_proxy = [(1 - l) * 100 for l in losses]  # Convert loss to proxy accuracy
        plt.plot(accuracy_proxy, label=f"SNR={snr}dB")
    plt.title(f"Training Convergence of LSTM - {case}")
    plt.xlabel("Training Epoch")
    plt.ylabel("Sensing Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Plot: ROC Curves
for case in ["Case 1", "Case 2"]:
    plt.figure(figsize=(7, 5))
    for snr in SNR_LIST:
        fpr, tpr = roc_per_case[case][snr]
        plt.plot(fpr * 100, tpr * 100, label=f"SNR={snr}dB")
    plt.title(f"ROC Curve - {case}")
    plt.xlabel("Probability of false alarm (%)")
    plt.ylabel("Probability of detection (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === PD vs SNR (at PFA = 5%) Plot ===
from sklearn.metrics import roc_curve

FIXED_PFA = 0.05  # 5% PFA
pd_case1 = []
pd_case2 = []
valid_snr_list = []

for snr in SNR_LIST:
    file_path = DATA_DIR / f"Data_SNR{snr}vol20.pth"
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        continue

    db = torch.load(file_path, map_location=DEVICE)
    test_data = db['testing data list']
    test_labels = db['testing label list']
    test_loader = DataLoader(SyntheticPSDCaseDataset(test_data, test_labels), batch_size=BATCH_SIZE)

    for case_name, loader in zip(["Case 1", "Case 2"],
                                 [full_train_loader, small_train_loader]):

        # Train model
        model = LSTMDetector().to(DEVICE)
        loss_fn = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(5):  # Smaller for faster PD estimation
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Get predictions for test set
        y_true, y_scores = evaluate_model(model, test_loader)
        fpr, tpr, thresholds = roc_curve(y_true.ravel(), y_scores.ravel())

        # Get PD when PFA ~ 5%
        idx = np.argmin(np.abs(fpr - FIXED_PFA))
        pd_value = tpr[idx] * 100  # to percentage

        if case_name == "Case 1":
            pd_case1.append(pd_value)
        else:
            pd_case2.append(pd_value)

    valid_snr_list.append(snr)

# Plotting
plt.figure()
plt.plot(valid_snr_list, pd_case1, marker='o', label='LSTM (Case 1)')
plt.plot(valid_snr_list, pd_case2, marker='s', label='LSTM (Case 2)')
plt.title("PD vs SNR (PFA = 5%)")
plt.xlabel("SNR (dB)")
plt.ylabel("Probability of Detection (%)")
plt.grid(True)
plt.legend()
plt.show()

