import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Create a synthetic dataset
np.random.seed(42)  # For reproducibility
n_samples = 100
n_features = 5

# Generate random data
data = np.random.rand(n_samples, n_features) * 10  # Scale data between 0 and 10
df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(n_features)])

# Step 2: Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Step 3: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
principal_components = pca.fit_transform(data_scaled)

# Create a DataFrame for the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 4: Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.7)
plt.title('PCA of Synthetic Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Print explained variance
print('Explained Variance Ratio:', pca.explained_variance_ratio_)

