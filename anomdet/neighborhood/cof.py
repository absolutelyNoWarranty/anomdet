# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import squareform, pdist
from sklearn.neighbors import NearestNeighbors

from ..base import BaseAnomalyDetector
from ..utils import check_n_neighbors


class COF(BaseAnomalyDetector):
    """Connectivity-based Outlier Factor

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
    "Enhancing Effectiveness of Outlier Detections for Low Density Patterns"
    Authors: Jian Tang, Zhixiang Chen, Ada Wai-chee Fu, David W. Cheung
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
        """Calculate connectivity-based outlier factor for each sample in X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        cof : array, shape (n_samples,)
            Connectivity-based outlier factor for each sample.
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
        
        ac_dist = np.zeros((num_rows, 1));
        for i in range(num_rows):
            neigh_ind = np.append(i, indices[i, :])
            dists = squareform(pdist(X[neigh_ind, :]))
            sbn_path = [0]
            out = np.arange(1, self.k+1)  # out : the points "outside" the sbn_path
            sbn_cost = np.zeros((1, self.k))
            
            for j in range(self.k):
                # distance from sbn_path to points outside
                dist_path_out = dists[np.ix_(sbn_path, out)]  
                min_cost = np.min(dist_path_out)
                sbn_cost[0, j] = min_cost
                
                # min_X min_Y : coord of the minimum element 
                # in the matrix dist_path_out
                idx = np.argmin(dist_path_out)
                min_X, min_Y = idx/dist_path_out.shape[1], idx%dist_path_out.shape[1]
                
                # add to the sbn_path and remove it from the set of points to consider
                sbn_path.append(out[min_Y])
                out = np.append(out[:min_Y], out[(min_Y+1):])

            weight_vec = np.arange(self.k, 0, -1)
            ac_dist[i] = np.dot(sbn_cost, weight_vec)*2/self.k/(self.k+1)
        
        cof = np.zeros(num_rows);
        for i in range(num_rows):
            cof[i] = self.k * ac_dist[i] / sum(ac_dist[indices[i]]);
        
        return(cof)
