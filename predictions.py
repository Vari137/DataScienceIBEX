import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib.dates import DateFormatter

import torch.nn as nn
import matplotlib.pyplot as plt


# Load and prepare data
try:
    def process_data(file_path):
        # Read CSV and parse dates
        df = pd.read_csv(file_path)
        dates = pd.to_datetime(df.iloc[:, 0], format='%d.%m.%Y')
        numeric_columns = df.iloc[:, 1:5]  # Get columns 1,2,3,4
        
        # Convert string numbers to float
        for col in numeric_columns.columns:
            numeric_columns[col] = (numeric_columns[col].astype(str)
                                  .str.replace('.', '')
                                  .str.replace(',', '.')
                                  .astype(float))
        return numeric_columns, dates

    train_data, train_dates = process_data('datos_entrenamiento.csv')
    val_data, val_dates = process_data('datos_validacion.csv')
    
except Exception as e:
    print(f"Error loading/processing data: {e}")
    exit()

# Debug print
print("Available columns in training data:", train_data.columns.tolist())

# Custom dataset class
class StockDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        self.data = torch.FloatTensor(data.values.astype(np.float32))
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
prediction_dates = val_dates[10:]  # Skip first 10 dates due to sequence_length

with torch.no_grad():
    for x, y in val_loader:
        x = x.reshape(x.shape[0], -1)
        pred = model(x)
        predictions.append(pred.numpy())
        actual_values.append(y.numpy())

predictions = np.array(predictions).reshape(-1, 4)
actual_values = np.array(actual_values).reshape(-1, 4)

# Plot results with dates
features = ['Opening', 'Close', 'Max', 'Min']
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(features):
    axes[idx].plot(prediction_dates, actual_values[:, idx], label='Actual')
    axes[idx].plot(prediction_dates, predictions[:, idx], label='Predicted')
    axes[idx].set_title(feature)
    axes[idx].legend()
    # Rotate x-axis labels for better readability
    axes[idx].tick_params(axis='x', rotation=45)
    # Format x-axis to show only day and month
    axes[idx].xaxis.set_major_formatter(DateFormatter('%d.%m.%y'))

plt.tight_layout()
plt.show()

