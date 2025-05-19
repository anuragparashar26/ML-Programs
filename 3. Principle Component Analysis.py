import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig

iris = load_iris()
iris_data = iris.data
iris_target = iris.target
iris_feature_names = iris.feature_names

df = pd.DataFrame(iris_data, columns=iris_feature_names)
df['Target'] = iris_target

example_data = iris_data[:5]
print("Example Data (First 5 Samples):")
print(example_data)

scaler = StandardScaler()
iris_data_scaled = scaler.fit_transform(iris_data)
example_data_scaled = scaler.transform(example_data)
print("\nStandardized Example Data:")
print(example_data_scaled)

n_samples = iris_data_scaled.shape[0]
mean_vector = np.mean(iris_data_scaled, axis=0)
X_centered = iris_data_scaled - mean_vector
cov_matrix_manual = (1 / (n_samples - 1)) * np.dot(X_centered.T,
X_centered)
print("\nManually Computed Covariance Matrix:")
print(cov_matrix_manual)

eigenvalues_manual, eigenvectors_manual = eig(cov_matrix_manual)
print("\nManually Computed Eigenvalues:")
print(eigenvalues_manual)
print("\nManually Computed Eigenvectors:")
print(eigenvectors_manual)

sorted_indices = np.argsort(eigenvalues_manual)[::-1]
top_2_indices = sorted_indices[:2]
top_2_eigenvectors = eigenvectors_manual[:, top_2_indices]
print("\nTop 2 Eigenvectors:")
print(top_2_eigenvectors)

iris_pca = np.dot(iris_data_scaled, top_2_eigenvectors)
example_pca = np.dot(example_data_scaled, top_2_eigenvectors)
print("\nReduced 2D Example Data:")
print(example_pca)

iris_pca_df = pd.DataFrame(data=iris_pca, columns=["Principal Component 1", "Principal Component 2"])
iris_pca_df['Target'] = iris_target

plt.figure(figsize=(8, 6))
sns.scatterplot(
x="Principal Component 1", y="Principal Component 2", hue="Target",
data=iris_pca_df,
palette="viridis", s=100, alpha=0.8
)
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Target", labels=iris.target_names)
plt.grid(alpha=0.5)
plt.show()