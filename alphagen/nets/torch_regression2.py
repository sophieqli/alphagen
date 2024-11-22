import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Generate synthetic dataset
# Let's create a linear relationship with 100 features for X and a target Y = sum(X, axis=1) + noise

torch.manual_seed(42)  # For reproducibility

num_points = 100000
num_X = 100
# Generate random data points: 100 features per sample, 100 samples
X = torch.rand(num_points, num_X) * 10  # 100 samples, each with 100 features (values between 0 and 10)

# Linear relationship: Y = sum(X, axis=1) + noise
Y = torch.sum(X, dim=1, keepdim=True) + torch.randn((num_points, 1)) * 2  # Y = sum(X) + noise



# 2. Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # input_dim features, 1 output

    def forward(self, x):
        return self.linear(x)

# 3. Instantiate the model with input_dim = 100
model = LinearRegressionModel(input_dim=num_X)

# 4. Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=1e-4)  # Stochastic Gradient Descent

# 5. Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing X to the model
    Y_pred = model(X)

    # Compute the loss
    loss = criterion(Y_pred, Y)

    # Backward pass: Compute gradients
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    # Save the loss value for later visualization
    losses.append(loss.item())

    print("\nFlattened Weights:")
    print(model.linear.weight.flatten())

    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print("\nFlattened Weights:")
print(model.linear.weight.flatten())

# 6. Plot the loss over epochs to visualize training progress
#plt.plot(range(num_epochs), losses)
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.title('Loss during training')
#plt.show()

# 7. Final learned parameters (weights and bias)
print(f'Learned weights: {model.linear.weight}')
print(f'Learned bias: {model.linear.bias.item():.4f}')

# 8. Make predictions
with torch.no_grad():
    Y_pred_final = model(X)

# 9. Since the model has 100 features, we can't visualize all of them easily.
# Instead, we will compare the predicted and actual sums of X (the target Y)

# Compare the predicted values (Y_pred_final) with the true values (Y)
print("Predicted Y vs Actual Y")
print("Predicted Y:", Y_pred_final[:10].flatten())  # Show the first 10 predictions
print("Actual Y:", Y[:10].flatten())  # Show the first 10 true values

