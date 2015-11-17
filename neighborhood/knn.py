# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..base import BaseAnomalyDetector
from ..utils import check_n_neighbors


class KNN(BaseAnomalyDetector):
    """Distance to nearest neighbors

    Note: Assumes none of the data are duplicated.
    
    Parameters
    ----------
    k : int or float, optional (default=1)
        
        The number of neighbors to use.
        
        If 
            int : 1 <= k < n_samples
                The exact number of neighbors
            float : Then 0.0 < k < 1.0
                Represents number of neighbors to use as a fraction of the
                total number of samples.

    """
    
    def __init__(self, k=1):
        self.k = k
    
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        y : unused parameter
        '''
        
        # Clear previous
        if hasattr(self, 'n_neighbors'):
            delattr(self, 'n_neighbors')
        if hasattr(self, 'nbrs'):
            delattr(self, 'nbrs')
        
        if X is not None:
            n = X.shape[0]
            self.n_neighbors = check_n_neighbors(self.k, n)
            self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(X)
        
        return self
    
    def predict(self, X, k=None):
        """Calculate KNN outlier factor for each sample in X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        k : int or float, optional (default=None)
        
            The number of neighbors to use for predict. If not given will use the default given
            in the construction.
            
            If 
                int : 1 <= k < n_samples
                    The exact number of neighbors
                float : Then 0.0 < k < 1.0
                    Represents number of neighbors to use as a fraction of the
                    total number of samples.
            
        Returns
        -------
        knn_outlier_factor : array, shape (n_samples,)
            KNN outlier factor for each sample.
        """
   
        if not hasattr(self, 'n_neighbors'):
            self.fit(X)
   
        if k is None:
            k = self.n_neighbors
        else:
            k = check_n_neighbors(k, X.shape[0])
                
        distances, _ = self.nbrs.kneighbors(X, n_neighbors=k+1)
        #distances = distances[:, 1:]
        distances[distances[:, 0] == 0., :-1] = distances[distances[:, 0] == 0., 1:]
        distances = distances[:, :-1]
        
        return distances.mean(axis=1)
