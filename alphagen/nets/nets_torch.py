import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Define the 2-layer Feed Forward Neural Network with Sigmoid Activation
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function
        self.layer2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid(x)  # Apply sigmoid activation
        x = self.layer2(x)
        return x

# Step 2: Create synthetic data (10000 data points, 10 features)
#np.random.seed(0)  # Set seed for reproducibility
#X_data = np.random.randn(10000, 10)  # 10000 samples, 10 features
#Y_data = np.sum(X_data, axis=1) + np.random.randn(10000) * 0.1  # Continuous target, sum of X + some noise

# Step 2: Create synthetic data (10000 data points, 10 features)
np.random.seed(0)  # Set seed for reproducibility
X_data = np.random.randn(10000, 10)  # 10000 samples, 10 features
Y_data = np.sum(X_data, axis=1, keepdims=True) + np.random.randn(10000, 1) * 0.1  # Continuous target, sum of X + some noise

# Normalize data (optional but often done for neural networks)
X_data = (X_data - np.mean(X_data, axis=0)) / np.std(X_data, axis=0)

# Convert data to torch tensors
X_tensor = torch.tensor(X_data, dtype=torch.float32)
Y_tensor = torch.tensor(Y_data, dtype=torch.float32).view(-1, 1)  # Reshape to be a column vector

# Step 3: Initialize the model, loss function, and optimizer
input_size = 10  # Number of features
hidden_size = 20  # Hidden layer size
output_size = 1  # Regression output

model = FeedForwardNN(input_size, hidden_size, output_size)

# Use Mean Squared Error loss function for regression
criterion = nn.MSELoss()

# Using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Neural Network
epochs = 1000  # Number of epochs for training
batch_size = 64  # Batch size for training

for epoch in range(epochs):
    # Shuffle the data for each epoch
    permutation = torch.randperm(X_tensor.size(0))
    X_tensor = X_tensor[permutation]
    Y_tensor = Y_tensor[permutation]

    # Train in batches
    for i in range(0, X_tensor.size(0), batch_size):
        # Get batch data
        X_batch = X_tensor[i:i+batch_size]
        Y_batch = Y_tensor[i:i+batch_size]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: Compute predicted y by passing x to the model
        Y_pred = model(X_batch)

        # Compute the loss
        loss = criterion(Y_pred, Y_batch)

        # Backward pass: Compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Evaluate the model on the training data
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode
    Y_pred_train = model(X_tensor)
    train_loss = criterion(Y_pred_train, Y_tensor)
    print(f"Final Training Loss: {train_loss.item():.4f}")

