import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Parameters
N = 10
epochs = 100

# Load and preprocess data
data = pd.read_csv("Datos Históricos del IBEX 35.csv")
data = data['Close'].values.reshape(-1, 1)  # Reshape for scaler

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Prepare data
def create_sequences(data, N):
    sequences = []
    targets = []
    for i in range(len(data) - N):
        sequences.append(data[i:i+N].flatten())  # Flatten the sequence
        targets.append(data[i+N].item())  # Get single value
    return np.array(sequences), np.array(targets)

sequences, targets = create_sequences(data, N)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    sequences, targets, test_size=0.2, random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(N, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss function and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    train_output = model(X_train)
    train_loss = criterion(train_output.squeeze(), y_train)
    train_loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output.squeeze(), y_val)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler
}, 'mlp_model.pth')