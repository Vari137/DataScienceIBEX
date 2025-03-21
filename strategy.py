import numpy as np
import matplotlib.pyplot as plt

def calculate_returns(predictions, actual_values):
    # Initialize variables to keep track of returns
    returns = []
    cumulative_returns = [1.0]
    
    # Iterate over the predictions and actual values
    for i in range(len(predictions)):
        # Calculate the return based on the model's prediction
        if predictions[i, 0] < actual_values[i, 0]:  # Buy signal
            return_val = actual_values[i, 3] / actual_values[i, 0]  # Max value divided by opening value
        else:  # Sell signal
            return_val = actual_values[i, 0] / actual_values[i, 3]  # Opening value divided by max value
        
        # Update the returns and cumulative returns
        returns.append(return_val)
        cumulative_returns.append(cumulative_returns[-1] * return_val)
    
    return np.array(returns), np.array(cumulative_returns)

def plot_returns(cumulative_returns):
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns)
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.show()

# Load the predictions and actual values
predictions = np.load('predictions.npy')
actual_values = np.load('actual_values.npy')

# Calculate the returns
returns, cumulative_returns = calculate_returns(predictions, actual_values)

# Plot the cumulative returns
plot_returns(cumulative_returns)