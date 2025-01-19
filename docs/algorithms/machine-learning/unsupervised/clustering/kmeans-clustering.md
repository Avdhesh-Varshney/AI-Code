# K Means Clustering 

**Overview:**  
K-means clustering is an unsupervised machine learning algorithm for grouping similar data points together into clusters based on their features.

---
### **Advantages of K-means:**  
- **Simple and Easy to implement**
- **Efficiency:** K-means is computationally efficient and can handle large datasets with high dimensionality.
- **Flexibility:** K-means offers flexibility as it can be easily customized for different applications, allowing the use of various distance metrics and 
   initialization techniques. 
- **Scalability:**  K-means can handle large datasets with many data points  

---
**How K-means Works (Scratch Implementation Guide):**  
### **Algorithm Overview:**
1. **Initialization**:
   - Choose `k` initial centroids randomly from the dataset.

2. **Iterative Process**:
   - **Assign Data Points**: For each data point, calculate the Euclidean distance to all centroids and assign the data point to the nearest centroid.
   - **Update Centroids**: Recalculate the centroids by averaging the data points assigned to each cluster.
   - **Check for Convergence**: If the centroids do not change significantly between iterations (i.e., they converge), stop. Otherwise, repeat the process.

3. **Termination**:
   - The algorithm terminates either when the centroids have converged or when the maximum number of iterations is reached.

4. **Output**:
   - The final cluster assignments for each data point. 


## Parameters

- `num_clusters`: Number of clusters to form.
- `max_iterations`: Maximum number of iterations before stopping.
- `show_steps`:	Whether to visualize the clustering process step by step (Boolean).

## Scratch Code 

- kmeans_scratch.py file 

```py
import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in space.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))

class KMeansClustering:
    def __init__(self, num_clusters=5, max_iterations=100, show_steps=False):
        """
        Initialize the KMeans clustering model with the following parameters:
        - num_clusters: Number of clusters we want to form
        - max_iterations: Maximum number of iterations for the algorithm
        - show_steps: Boolean flag to visualize the clustering process step by step
        """
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.show_steps = show_steps
        self.clusters = [[] for _ in range(self.num_clusters)]  # Initialize empty clusters
        self.centroids = []  # List to store the centroids of clusters

    def fit_predict(self, data):
        """
        Fit the KMeans model on the data and predict the cluster labels for each data point.
        """
        self.data = data
        self.num_samples, self.num_features = data.shape  # Get number of samples and features
        initial_sample_indices = np.random.choice(self.num_samples, self.num_clusters, replace=False)
        self.centroids = [self.data[idx] for idx in initial_sample_indices]
        
        for _ in range(self.max_iterations):
            # Step 1: Assign each data point to the closest centroid to form clusters
            self.clusters = self._assign_to_clusters(self.centroids)
            if self.show_steps:
                self._plot_clusters()

            # Step 2: Calculate new centroids by averaging the data points in each cluster
            old_centroids = self.centroids
            self.centroids = self._calculate_new_centroids(self.clusters)

            # Step 3: Check for convergence 
            if self._has_converged(old_centroids, self.centroids):
                break
            if self.show_steps:
                self._plot_clusters()

        return self._get_cluster_labels(self.clusters)

    def _assign_to_clusters(self, centroids):
        """
        Assign each data point to the closest centroid based on Euclidean distance.
        """
        clusters = [[] for _ in range(self.num_clusters)]
        for sample_idx, sample in enumerate(self.data):
            closest_centroid_idx = self._find_closest_centroid(sample, centroids)
            clusters[closest_centroid_idx].append(sample_idx)
        return clusters

    def _find_closest_centroid(self, sample, centroids):
        """
        Find the index of the closest centroid to the given data point (sample).
        """
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        closest_idx = np.argmin(distances)  # Index of the closest centroid
        return closest_idx

    def _calculate_new_centroids(self, clusters):
        """
        Calculate new centroids by averaging the data points in each cluster.
        """
        centroids = np.zeros((self.num_clusters, self.num_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _has_converged(self, old_centroids, new_centroids):
        """
        Check if the centroids have converged 
        """
        distances = [euclidean_distance(old_centroids[i], new_centroids[i]) for i in range(self.num_clusters)]
        return sum(distances) == 0  # If centroids haven't moved, they are converged

    def _get_cluster_labels(self, clusters):
        """
        Get the cluster labels for each data point based on the final clusters.
        """
        labels = np.empty(self.num_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _plot_clusters(self):
        """
        Visualize the clusters and centroids in a 2D plot using matplotlib.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, cluster in enumerate(self.clusters):
            cluster_points = self.data[cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1])

        for centroid in self.centroids:
            ax.scatter(centroid[0], centroid[1], marker="x", color="black", linewidth=2)

        plt.show()

```

- test_kmeans.py file 

```py
import unittest
import numpy as np
from kmeans_scratch import KMeansClustering 

class TestKMeansClustering(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.X_train = np.vstack([
            np.random.randn(100, 2) + np.array([5, 5]),
            np.random.randn(100, 2) + np.array([-5, -5]),
            np.random.randn(100, 2) + np.array([5, -5]),
            np.random.randn(100, 2) + np.array([-5, 5])
        ])
        
    def test_kmeans(self):
        """Test the basic KMeans clustering functionality"""
        kmeans = KMeansClustering(num_clusters=4, max_iterations=100, show_steps=False)
        
        cluster_labels = kmeans.fit_predict(self.X_train)
        
        unique_labels = np.unique(cluster_labels)
        self.assertEqual(len(unique_labels), 4)  
        self.assertEqual(cluster_labels.shape, (self.X_train.shape[0],))  
        print("Cluster labels for the data points:")
        print(cluster_labels)

if __name__ == '__main__':
    unittest.main()
