import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import matplotlib.pyplot as plt

# Custom dataset class
class StockDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.data = torch.FloatTensor(data[['opening', 'close', 'max', 'min']].values)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.sequence_length]
        y = self.data[idx+self.sequence_length]
        return x, y

# Model definition
class StockPredictor(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(40, 64),  # 10 days * 4 features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)    # 4 output features
        )

# Load and prepare data
train_data = pd.read_csv('datos_entrenamiento.csv')
val_data = pd.read_csv('datos_validacion.csv')

# Create datasets and dataloaders
train_dataset = StockDataset(train_data)
val_dataset = StockDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Initialize model, loss function and optimizer
model = StockPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x.reshape(batch_x.shape[0], -1))
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Validation and plotting
model.eval()
predictions = []
actual_values = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.reshape(x.shape[0], -1)
        pred = model(x)
        predictions.append(pred.numpy())
        actual_values.append(y.numpy())

predictions = np.array(predictions).reshape(-1, 4)
actual_values = np.array(actual_values).reshape(-1, 4)

# Plot results
features = ['Opening', 'Close', 'Max', 'Min']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(features):
    axes[idx].plot(actual_values[:, idx], label='Actual')
    axes[idx].plot(predictions[:, idx], label='Predicted')
    axes[idx].set_title(feature)
    axes[idx].legend()

plt.tight_layout()
plt.show()