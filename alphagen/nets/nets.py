import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases for layer 1 (input to hidden layer)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Random small values
        self.b1 = np.zeros((1, hidden_size))

        # Weights and biases for layer 2 (hidden to output layer)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):
        """Forward pass through the network."""
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear transformation for layer 1
        self.A1 = self.sigmoid(self.Z1)  # Apply sigmoid activation

        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear transformation for layer 2 (output layer)
        self.A2 = self.Z2  # No activation for regression output

        return self.A2

    def backward(self, X, Y, learning_rate=0.01):
        """Backward pass using gradient descent."""
        m = X.shape[0]  # Number of training examples

        # Compute the error at the output layer
        dZ2 = self.A2 - Y  # For regression, no activation at output, so simple error
        dW2 = np.dot(self.A1.T, dZ2) / m  # Gradient of W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Gradient of b2

        # Compute the error at the hidden layer
        dA1 = np.dot(dZ2, self.W2.T)  # Backpropagate error
        dZ1 = dA1 * self.sigmoid_derivative(self.Z1)  # Apply derivative of sigmoid
        dW1 = np.dot(X.T, dZ1) / m  # Gradient of W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Gradient of b1

        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, Y, epochs=1000, learning_rate=0.01, batch_size=64):
        """Train the network on the provided data."""
        for epoch in range(epochs):
            # Shuffle the data for each epoch
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            # Train in batches
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]

                # Forward pass
                Y_pred = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, Y_batch, learning_rate)

            # Print the loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                Y_pred = self.forward(X)
                loss = np.mean((Y_pred - Y) ** 2)  # Mean Squared Error
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}')

    def predict(self, X):
        """Predict the output for given input X."""
        return self.forward(X)


# Step 2: Create synthetic data (10000 data points, 10 features)
np.random.seed(0)  # Set seed for reproducibility
X_data = np.random.randn(10000, 10)  # 10000 samples, 10 features
Y_data = np.sum(X_data, axis=1, keepdims=True) + np.random.randn(10000, 1) * 0.1  # Continuous target, sum of X + some noise

# Normalize data (optional but often done for neural networks)
X_data = (X_data - np.mean(X_data, axis=0)) / np.std(X_data, axis=0)



# Step 3: Initialize and train the model
input_size = 10  # Number of features
hidden_size = 20  # Hidden layer size
output_size = 1  # Regression output

# Initialize the neural network
model = NeuralNetwork(input_size, hidden_size, output_size)

# Train the network
model.train(X_data, Y_data, epochs=1000, learning_rate=0.01, batch_size=64)

# After training, evaluate the model
Y_pred_train = model.predict(X_data)
train_loss = np.mean((Y_pred_train - Y_data) ** 2)
print(f"Final Training Loss: {train_loss:.4f}")

