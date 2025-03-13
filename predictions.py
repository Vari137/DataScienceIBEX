import torch
import pandas as pd

import torch.nn as nn
import torch.optim as optim

# Assuming datos_entrenamiento and datos_validacion are pandas DataFrames

# Load your data
datos_entrenamiento = pd.read_csv('datos_entrenamiento.csv')
datos_validacion = pd.read_csv('datos_validacion.csv')

# Prepare the data
X_train = datos_entrenamiento.iloc[:, 1:-1].values
y_train = datos_entrenamiento.iloc[:, -1].values
X_val = datos_validacion.iloc[:, 1:-1].values
y_val = datos_validacion.iloc[:, -1].values

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Predict the next row
model.eval()
with torch.no_grad():
    next_row_prediction = model(X_val[-1].view(1, -1))
    print(f'Next row prediction: {next_row_prediction.item():.4f}')