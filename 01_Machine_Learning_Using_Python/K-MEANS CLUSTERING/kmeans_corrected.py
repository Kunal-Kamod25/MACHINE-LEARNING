import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Get dataset dimensions
n = data.shape[0]  # 150 samples
m = data.shape[1]  # 4 features
k = 3  # Number of clusters (NOT n*m!)

print(f"Dataset: {n} samples, {m} features")
print(f"Number of clusters: {k}")

# Use the full dataset
X = data.values

# Distance calculation function
def calc_distance(x1, x2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Initialize centroids randomly
random_index = np.random.choice(len(X), k, replace=False)
centroids = X[random_index]

print(f"\nInitial centroids indices: {random_index}")

# Calculate distances from each point to each centroid
distance0 = []
distance1 = []
distance2 = []

for i in range(n):
    dist = calc_distance(X[i], centroids[0])
    distance0.append(dist)
    
for i in range(n):
    dist = calc_distance(X[i], centroids[1])
    distance1.append(dist)

for i in range(n):
    dist = calc_distance(X[i], centroids[2])
    distance2.append(dist)

# Assign labels based on minimum distance
# FIXED: np.argmin needs an array/list, not separate arguments!
labels = []
for i in range(n):
    # Pass distances as a list to np.argmin
    labels.append(np.argmin([distance0[i], distance1[i], distance2[i]]))

print(f"\nNumber of labels assigned: {len(labels)}")
print(f"Unique labels: {np.unique(labels)}")
print(f"Label distribution: {np.bincount(labels)}")

# Show first 10 labels
print(f"\nFirst 10 labels: {labels[:10]}")
