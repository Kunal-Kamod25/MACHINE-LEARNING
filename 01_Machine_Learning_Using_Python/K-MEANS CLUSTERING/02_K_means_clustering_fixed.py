import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# Use the FULL dataset (all 150 samples) - THIS WAS THE BUG!
X = data.values  # Fixed: was data.sample(n=2, random_state=42).values

# Set parameters
k = 3  # number of clusters
n = 150  # number of samples (must match X.shape[0])

# Distance calculation function
def calc_distance(x1, x2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Initialize centroids randomly
random_index = np.random.choice(len(X), k, replace=False)
centroids = X[random_index]

print(f"Initial centroids shape: {centroids.shape}")
print(f"Dataset shape: {X.shape}")

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
    dist = calc_distance(X[i], centroids[2])  # Fixed: was centroids[1]
    distance2.append(dist)

# Assign labels based on minimum distance
labels = []
for i in range(n):
    # Find which centroid is closest
    distances = [distance0[i], distance1[i], distance2[i]]
    labels.append(np.argmin(distances))

print(f"\nNumber of labels assigned: {len(labels)}")
print(f"Unique labels: {np.unique(labels)}")
print(f"Label distribution: {np.bincount(labels)}")
