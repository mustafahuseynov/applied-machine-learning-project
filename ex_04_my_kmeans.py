from typing import Literal
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from dtaidistance import dtw, dtw_ndim

DISTANCE_METRICS = Literal["euclidean", "manhattan", "dtw"]
INIT_METHOD = Literal["random", "kmeans++"]

class MyKMeans:
    """
    Custom K-means clustering implementation with support for multiple distance metrics.
    
    Args:
        k (int): Number of clusters.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        distance_metric (str, optional): Distance metric to use. Options are "euclidean", 
                                         "manhattan", or "dtw". Defaults to "euclidean".
        init_method (str, optional): Initialization method to use. Options are "kmeans++" or "random". Defaults to "kmeans++".
    """
    def __init__(
        self,
        k: int,
        max_iter: int = 100,
        distance_metric: DISTANCE_METRICS = "euclidean",
        init_method: INIT_METHOD = "kmeans++",
        random_state: int | None = None,
        tol: float = 1e-4
    ):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        if distance_metric not in ("euclidean", "manhattan", "dtw"):
            raise ValueError("Unsupported distance metric")

        if init_method not in ("random", "kmeans++"):
            raise ValueError("Unsupported initialization method")

        self.k = k
        self.max_iter = max_iter
        self.distance_metric = distance_metric
        self.init_method = init_method
        self.tol = tol

        self.centroids = None
        self.inertia_ = None
        self.labels_ = None
        self.n_iter_ = 0

        self._rng = np.random.default_rng(random_state)

    def _validate_data(self, x):
        """Convert input to a valid numerical NumPy array."""

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        elif not isinstance(x, np.ndarray):
            raise ValueError(
                "Input data must be a numpy array or a pandas DataFrame"
            )

        x = np.asarray(x, dtype=np.double)

        if x.ndim not in (2, 3):
            raise ValueError("Input data must be a 2D or 3D array")

        if x.shape[0] == 0:
            raise ValueError("Input data cannot be empty")

        if any(dim == 0 for dim in x.shape[1:]):
            raise ValueError("Input dimensions cannot be empty")

        if not np.all(np.isfinite(x)):
            raise ValueError("Input data must contain only finite values")

        return np.ascontiguousarray(x)
    
    def fit(self, x: np.ndarray | pd.DataFrame):
        """
        Fit the K-means model to the data.
        
        Args:
            x (np.ndarray | pd.DataFrame): Training data of shape (n_samples, n_features).
        
        Returns:
            MyKMeans: Fitted estimator instance.
        """

        # Convert pandas DataFrame to NumPy array

        x = self._validate_data(x)

        if x.shape[0] < self.k:
            raise ValueError("Number of samples must be at least k")

        # Initialize centroids
        self.centroids = self._initialize_centroids(x)

        n_samples = x.shape[0]
        self.n_iter_ = 0

        for iteration in tqdm(range(self.max_iter)):
            distances = self._compute_distance(x, self.centroids)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.empty_like(self.centroids)

            nearest_distances = distances[
                np.arange(n_samples), labels
            ]

            farthest_indices = np.argsort(nearest_distances)[::-1]
            used_indices = set()

            for cluster in range(self.k):
                cluster_points = x[labels == cluster]

                if len(cluster_points) > 0:
                    new_centroids[cluster] = np.mean(
                        cluster_points,
                        axis=0
                    )
                else:
                    for index in farthest_indices:
                        if index not in used_indices:
                            new_centroids[cluster] = x[index]
                            used_indices.add(index)
                            break

            centroid_shift = np.linalg.norm(
                (new_centroids - self.centroids).reshape(self.k, -1)
            )

            self.centroids = new_centroids
            self.n_iter_ = iteration + 1

            if centroid_shift <= self.tol:
                break

        # Calculate final labels and inertia
        final_distances = self._compute_distance(x, self.centroids)
        self.labels_ = np.argmin(final_distances, axis=1)

        self.inertia_ = np.sum(
            final_distances[np.arange(n_samples), self.labels_]
        )

        return self

    def fit_predict(self, x: np.ndarray):
        """
        Fit the K-means model to the data and return the predicted labels.
        """

        self.fit(x)
        return self.predict(x)

    def predict(self, x: np.ndarray):
        """
        Predict the closest cluster for each sample in x.
        
        Args:
            x (np.ndarray): New data to predict, of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Index of the cluster each sample belongs to.
        """

        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")

        x = self._validate_data(x)

        if x.ndim != self.centroids.ndim:
            raise ValueError(
                "Input dimensions must match the training data"
            )

        if self.distance_metric == "dtw":

            # For DTW, the number of features must match.
            if x.ndim == 3 and x.shape[2] != self.centroids.shape[2]:
                raise ValueError(
                    "Number of features must match the training data"
                )

        else:

            # Euclidean and Manhattan require matching dimensions.
            if x.shape[1:] != self.centroids.shape[1:]:
                raise ValueError(
                    "Input dimensions must match the training data"
                )

        # Compute distances between samples and centroids
        distances = self._compute_distance(x, self.centroids)
        
        # Return the index of the closest centroid for each sample
        return np.argmin(distances, axis=1)

    def _initialize_centroids(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centroids using random selection or K-means++.
        """

        n_samples = x.shape[0]

        # Random initialization
        if self.init_method == "random":
            indices = self._rng.choice(
                n_samples,
                size=self.k,
                replace=False
            )

            return x[indices].copy()

        # K-means++ initialization
        selected_indices = []

        # Select the first centroid randomly
        first_index = self._rng.integers(n_samples)

        selected_indices.append(first_index)

        centroids = [x[first_index].copy()]

        # Select the remaining centroids
        for _ in range(1, self.k):

            current_centroids = np.asarray(centroids)

            distances = self._compute_distance(
                x,
                current_centroids
            )

            # Distance to the nearest existing centroid
            closest_distances = np.min(distances, axis=1)

            # K-means++ uses squared distances as probabilities
            probabilities = closest_distances ** 2

            # Do not select an already selected sample
            probabilities[selected_indices] = 0

            total = np.sum(probabilities)

            if total == 0 or not np.isfinite(total):

                remaining_indices = np.setdiff1d(
                    np.arange(n_samples),
                    selected_indices
                )

                next_index = self._rng.choice(remaining_indices)

            else:

                probabilities /= total

                next_index = self._rng.choice(
                    n_samples,
                    p=probabilities
                )

            selected_indices.append(next_index)
            centroids.append(x[next_index].copy())

        return np.asarray(centroids)

    def _compute_distance(
        self,
        x: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """
        Calculate distances between samples and centroids.

        Returns:
            Array with shape (n_samples, k).
        """

        if self.distance_metric == "dtw":
            return self._dtw(x, centroids)

        # Flatten all dimensions except the sample dimension.
        x_flat = x.reshape(x.shape[0], -1)
        centroid_flat = centroids.reshape(centroids.shape[0], -1)

        differences = (
            x_flat[:, np.newaxis, :] -
            centroid_flat[np.newaxis, :, :]
        )

        if self.distance_metric == "euclidean":

            return np.sqrt(
                np.sum(differences ** 2, axis=2)
            )

        elif self.distance_metric == "manhattan":

            return np.sum(
                np.abs(differences),
                axis=2
            )

        raise ValueError("Unsupported distance metric")
    
    def _dtw(self, x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Simplified DTW distance computation using dtaidistance.
        
        Args:
            x (np.ndarray): Data points of shape (n_samples, time_steps, n_features) or (n_samples, n_features)
            centroids (np.ndarray): Centroids of shape (k, time_steps, n_features) or (k, n_features)
            
        Returns:
            np.ndarray: DTW distances between each sample and each centroid, shape (n_samples, k).
        """
        n_samples = x.shape[0]
        n_centroids = centroids.shape[0]

        distances = np.zeros(
            (n_samples, n_centroids),
            dtype=np.double
        )

        if x.ndim == 2:

            # Univariate DTW
            for i in range(n_samples):
                for j in range(n_centroids):

                    distances[i, j] = dtw.distance(
                        x[i],
                        centroids[j],
                        use_c=True
                    )

        elif x.ndim == 3:

            # Multivariate DTW
            for i in range(n_samples):
                for j in range(n_centroids):

                    distances[i, j] = dtw_ndim.distance(
                        x[i],
                        centroids[j],
                        use_c=True
                    )

        return distances
