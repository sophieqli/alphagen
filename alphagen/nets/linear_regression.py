import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 

# 1. Generate synthetic dataset
# Let's create a linear relationship with 100 features for X and a target Y = sum(X, axis=1) + noise

torch.manual_seed(42)  # For reproducibility

num_points = 1000
num_X = 100
# Generate random data points: 100 features per sample, 100 samples
X = torch.rand(num_points, num_X) * 10  # 100 samples, each with 100 features (values between 0 and 10)
#some tensor indexing + agg functions
#print(X[0:2,0:2])
#print(torch.mean(X, 1).shape, torch.max(X), torch.std(X))

# Linear relationship: Y = sum(X, axis=1) + noise
Y = torch.sum(X, dim=1, keepdim=True) + torch.randn((num_points, 1)) * 2  # Y = sum(X) + noise
print(torch.sum(X, dim = 1, keepdim = True), torch.sum(X, dim = 1, keepdim = True).shape)


inputs = [[1,2,3],[4,5,6]]
#print(torch.tensor(inputs.to_numpy))

######### Now using linear regression close form formula 
X_b = torch.cat([torch.ones(num_points, 1), X], dim=1)  # Add the bias term
# 3. Compute the weights using the normal equation: w = (X^T X)^(-1) X^T y
X_transpose = X_b.T  # Transpose of X_b
print("CONCAT ", X_b.shape, X_transpose.shape)
theta = torch.linalg.inv(X_transpose @ X_b) @ X_transpose @ Y  # Normal equation

# 4. Print out the learned weights (theta values)
print("Learned weights (including bias):", theta.T.shape)
print(theta.T)

# 5. Make predictions (X_new can be any new data point, here we use X_b)
Y_pred = X_b @ theta  # Predictions
#makes sense given we took sum, so coef shld be ~1

############# use gradient descent method to solve it and compare the solution
# TASK1 TODO: write the code for it and compare the solution


#below is stoch. grad des but same idea as regular
def linloss(w, i): 
    #i is the entry 
    pred = X_b[i] @ w.T #take their dot prod  
    return (pred - Y[i])**2
def lingrad(w, i): 
    pred = X_b[i] @ w.T #take their dot prod  
    return 2*(pred - Y[i])*X_b[i]
def SGD(loss, grad, w_i):
    w = w_i
    eta = 0.0001
    for t in range(100): 
        for i in range(num_points): #shld be len(x) 
            w -= eta*lingrad(w,i) 
        print("t = ", t, "w = ", w, "  Loss = ", loss(w, i)) #" gradientfunction = ", lingrad(w,i))
         
SGD(linloss, lingrad, torch.zeros(1, num_X+1))







################# use stochastic gradient descent method to solve it and compare the solution
# TASK2: TODO: write the code for it and compare the solution




