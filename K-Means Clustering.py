import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data=load_breast_cancer()
X=data.data
y=data.target

feature_names=data.feature_names
target_names=data.target_names

print('Data Shape: ', X.shape)
print('Target: ', target_names)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
kmeans=KMeans(n_clusters=2, random_state=42, n_init=10)
clusters=kmeans.fit_predict(X_scaled)
labels_mapped=np.where(clusters==0,1,0)

print("Accuracy Score: ", accuracy_score(y, labels_mapped))
print("Confusion Matrix: \n", confusion_matrix(y, labels_mapped))

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=250, label='Centroids')
plt.xlabel('PCA COMP 1')
plt.ylabel('PCA COMP 2')
plt.title('Data')
plt.legend()
plt.grid(True)
plt.show()