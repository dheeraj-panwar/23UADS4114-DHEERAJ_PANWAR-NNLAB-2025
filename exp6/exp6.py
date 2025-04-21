import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the milk production dataset (has 2 columns: 'Month' and 'Production')
data = pd.read_csv('milk.csv')

# Check actual column names
print(data.columns)  # Should show: ['Month', 'Production']

# Rename column if needed
data.columns = ['Month', 'Milk_Production']

# Drop NA and use only the production values
data.dropna(inplace=True)
milk_values = data[['Milk_Production']].values

# Normalize
scaler = MinMaxScaler(feature_range=(0, 1))
milk_values = scaler.fit_transform(milk_values)

# Create sequences
def create_dataset(series, seq_length):
    X, y = [], []
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12
X, y = create_dataset(milk_values, seq_length)

# Convert to PyTorch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
X = X.view(X.size(0), seq_length, 1)

# Define RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Model setup
input_size = 1
hidden_size = 128
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    output = model(X)
    optimizer.zero_grad()
    loss = criterion(output.squeeze(), y.squeeze())
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Accuracy calculation
    predicted = output.detach().numpy()
    actual = y.detach().numpy()
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(actual)
    mae = np.mean(np.abs(predicted.flatten() - actual.flatten()))
    mean_actual = np.mean(actual.flatten())
    accuracy = 1 - (mae / mean_actual)
    accuracies.append(accuracy * 100)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy*100:.2f}%")

# Final evaluation
model.eval()
predicted = model(X).detach().numpy()
actual = y.detach().numpy()
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(actual)
final_loss = np.mean((predicted.flatten() - actual.flatten()) ** 2)
final_mae = np.mean(np.abs(predicted.flatten() - actual.flatten()))
final_mean_actual = np.mean(actual.flatten())
final_accuracy = 1 - (final_mae / final_mean_actual)

print(f"\nFinal MSE Loss: {final_loss:.4f}")
print(f"Final Custom Accuracy: {final_accuracy*100:.2f}%")

# Plotting
plt.figure(figsize=(14, 8))
plt.subplot(3, 1, 1)
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted')
plt.title('Milk Production Forecasting')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(losses, label='Loss', color='red')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(accuracies, label='Accuracy (%)', color='green')
plt.title('Accuracy per Epoch')
plt.legend()

plt.tight_layout()
plt.show()

