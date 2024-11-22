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


######### Now using linear regression close form formula 
#X_b = torch.cat([torch.ones(num_points, 1), X], dim=1)  # Add the bias term
X_b = X
# 3. Compute the weights using the normal equation:
# w = (X^T X)^(-1) X^T y
X_transpose = X_b.T  # Transpose of X_b
theta = torch.linalg.inv(X_transpose @ X_b) @ X_transpose @ Y  # Normal equation

# 4. Print out the learned weights (theta values)
print("Learned weights (including bias):")
print(theta.T)

# 5. Make predictions (X_new can be any new data point, here we use X_b)
Y_pred = X_b @ theta  # Predictions



############# use gradient descent method to solve it and compare the solution
# TASK1 TODO: write the code for it and compare the solution








################# use stochastic gradient descent method to solve it and compare the solution
# TASK2: TODO: write the code for it and compare the solution




