import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing(as_frame=True)
data = housing_data['data']
print(data)
data['MedHouseVal'] = housing_data['target']

print("Creating histograms for all numerical features...")
for column in data.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(data[column], bins=50, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

print("Creating box plots for all numerical features to identify outliers...")
for column in data.columns:
    plt.figure(figsize=(8, 5))
    plt.boxplot(data[column], vert=False, patch_artist=True,
    boxprops=dict(facecolor='skyblue', color='blue'))
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

print("Identifying potential outliers using the IQR method...")
outliers = {}
for column in data.columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    print(f"{column}:")
    print(f"Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    print(f"Number of outliers: {len(outliers[column])}")
    print("---")
