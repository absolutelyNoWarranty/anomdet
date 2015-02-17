# -*- coding: utf-8 -*-

from .base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np

class KNN(BaseAnomalyDetector):
    """Simple sum of distances to nearest neighbors heuristic

    Parameters
    ----------

    k : int
        Number of nearest neighbors to use

    Note: There are like a billion different tiny, adjustments you can make to this basic idea. F*** that noise,
    I'm not going to include them here.
    """
    
    def __init__(self, k):
        self.k = k
    
    def predict(self, X, k=None):
        """Calculate KNN outlier factor for each sample in X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        knn_outlier_factor : array, shape (n_samples,)
            KNN outlier factor for each sample.
        """
        # If reference_X exists use it, otherwise use X itself
        if self.reference_X is None:
            self.reference_X = X
            self.nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(self.reference_X)
        if k is None:
            k = self.k
        distances, indices = self.nbrs.kneighbors(X, n_neighbors=k+1)
        distances = distances[:, 1:]
        
        return distances.mean(axis=1)
    
    def fit(self, X=None, y=None):
        if self.k <= 0 or not isinstance(self.k, int):
            raise ValueError("k needs to be a positive integer.")
        self.reference_X = X
        if X is not None:
            nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(self.reference_X)
            self.nbrs = nbrs
        else:
            self.nbrs = None
        return self