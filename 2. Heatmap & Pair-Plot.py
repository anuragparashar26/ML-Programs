import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing(as_frame=True)
data = housing_data['data']
data['MedHouseVal'] = housing_data['target'] 

print("Computing the correlation matrix...")
correlation_matrix = data.corr()
print(correlation_matrix)

print("Visualizing the correlation matrix using a heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm",
cbar=True, square=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

print("Creating a pair plot to visualize pairwise relationships between features...")
sns.pairplot(data, diag_kind='kde', corner=True)
plt.show()