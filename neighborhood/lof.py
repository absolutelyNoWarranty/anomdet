# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..base import BaseAnomalyDetector
from ..utils import check_n_neighbors


class LOF(BaseAnomalyDetector):
    """Calculate Local Outlier Factor of data

    Parameters
    ----------
    k : int or float, optional (default=5)
        
        The number of neighbors to use.
        
        If 
            int : 1 <= k < n_samples
                The exact number of neighbors
            float : Then 0.0 < k < 1.0
                Represents number of neighbors to use as a fraction of the
                total number of samples.

    References
    ----------
    Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000. LOF: identifying density-based local outliers. SIGMOD Rec. 29, 2 (May 2000), 93-104. DOI=10.1145/335191.335388 http://doi.acm.org/10.1145/335191.335388
    """
    
    def __init__(self, k=5):
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
        
    def predict(self, X):
        """Calculate local outlier factor for each sample in X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        lof : array, shape (n_samples,)
            Local outlier factor for each sample.
        """
        if not hasattr(self, 'n_neighbors'):
            self.fit(X)
        
        nbrs = self.nbrs
        distances, indices = nbrs.kneighbors(X)
        
        exists_dupl = distances[:, 0] == 0.
        distances[exists_dupl, :-1] = distances[exists_dupl, 1:]
        indices[exists_dupl, :-1] = indices[exists_dupl, 1:]
        distances = distances[:, :-1]
        indices = indices[:, :-1]
        
        k_dists = distances[:, -1]
        
        num_rows = X.shape[0]
        
        lrd_value = np.zeros((num_rows, 1));
        lrd_value = np.zeros(num_rows)
        for i in xrange(num_rows):
            temp = X[i, :] - X[indices[i], :]
            temp = np.sqrt(np.sum(temp**2, 1))
            reachability_dists = np.max(np.vstack([temp, k_dists[indices[i]]]), 0)
            lrd_value[i] = self.k/sum(reachability_dists);
        
        lof = np.zeros(num_rows)
        for i in xrange(num_rows):
            lof[i] = np.sum(lrd_value[indices[i]]) / lrd_value[i] / self.k
        
        return(lof)