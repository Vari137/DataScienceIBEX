import yfinance as yf
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib.dates import DateFormatter

import torch.nn as nn
import matplotlib.pyplot as plt


# Load and prepare data from Yahoo Finance
def process_data_yahoo(ticker, start_date, end_date):
    # Fetch data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    dates = pd.to_datetime(df.index)
    
    # Select relevant columns: Open, High, Low, Close
    numeric_columns = df[['Open', 'High', 'Low', 'Close']].copy()
    return numeric_columns, dates

ticket = '^IBEX'
# Train and validation data
try:
    train_data, train_dates = process_data_yahoo(ticket, '2010-01-01', '2022-12-31')
    val_data, val_dates = process_data_yahoo(ticket, '2023-01-01', '2024-12-31')
except Exception as e:
    print(f"Error fetching/processing data: {e}")
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
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
# Export model
torch.save(model.state_dict(), 'stock_predictor.pth')


# Create two separate figures
# Figure 1: Original validation plots
features = ['Opening', 'Close', 'Max', 'Min']
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))
axes1 = axes1.ravel()

for idx, feature in enumerate(features):
    axes1[idx].plot(prediction_dates, actual_values[:, idx], label='Actual')
    
    # Find local maximum and minimums
    diff = np.diff(predictions[:, idx])
    max_idx = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0] + 1
    min_idx = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0] + 1
    
    # Plot local maximum and minimums as buy and sell signals
    axes1[idx].scatter(prediction_dates[max_idx], actual_values[max_idx, idx], label='Sell', color='r', marker='v')
    axes1[idx].scatter(prediction_dates[min_idx], actual_values[min_idx, idx], label='Buy', color='g', marker='^')
        
    axes1[idx].set_title(f'Validation Period - {feature}')
    axes1[idx].legend()
    axes1[idx].tick_params(axis='x', rotation=45)
    axes1[idx].xaxis.set_major_formatter(DateFormatter('%d.%m.%y'))

plt.figure(1)
plt.tight_layout()

# Calculate cumulative returns
cumulative_returns = [1.0]
buy_signals = prediction_dates[min_idx]
sell_signals = prediction_dates[max_idx]

for i in range(len(buy_signals) - 1):
    buy_price = actual_values[min_idx[i], 0]  # Buy at opening price
    sell_price = actual_values[max_idx[i], 0] 
    return_val = sell_price / buy_price
    cumulative_returns.append(cumulative_returns[-1] * return_val)

# Plot cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns)
plt.title('Cumulative Returns')
plt.xlabel('Time')
plt.ylabel('Return')
plt.show()

# Figure 2: Historical + Validation comparison
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10))
axes2 = axes2.ravel()

for idx, feature in enumerate(features):
    # Plot historical data (2010-2022)
    axes2[idx].plot(train_dates, train_data.iloc[:, idx], 
                   'b-', label='Historical (2010-2022)')
    
    # Plot validation period (2022-2024)
    axes2[idx].plot(prediction_dates, actual_values[:, idx], 
                   'g-', label='Actual (2022-2024)')
    axes2[idx].plot(prediction_dates, predictions[:, idx], 
                   'r--', label='Predicted (2022-2024)')
    
    axes2[idx].set_title(f'Historical + Validation - {feature}')
    axes2[idx].legend()
    axes2[idx].tick_params(axis='x', rotation=45)
    axes2[idx].xaxis.set_major_formatter(DateFormatter('%y'))
    
    # Add vertical line at the transition point
    axes2[idx].axvline(x=train_dates[-1], color='gray', 
                      linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Save plots as PNGs
fig1.savefig(f'{ticket}_validation_period_plots.png', format='png', bbox_inches='tight')
plt.figure(2)
plt.savefig(f'{ticket}_cumulative_returns.png', format='png', bbox_inches='tight')
fig2.savefig(f'{ticket}_historical_validation_comparison.png', format='png', bbox_inches='tight')

