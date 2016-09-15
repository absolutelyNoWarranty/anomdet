# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import erf
from sklearn.neighbors import NearestNeighbors

from ..base import BaseAnomalyDetector
from ..utils import check_n_neighbors

class LoOP(BaseAnomalyDetector):
    """LoOP : Local Outlier Probabilites

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
                
    lambda_ : float, optional (default=3.0)
        Scaling parameter for the outlier probabilities.

    References
    ----------
    Kriegel, Hans-Peter, et al. "LoOP: local outlier probabilities." Proceedings of the 18th ACM conference on Information and knowledge management. ACM, 2009.
    """
    
    def __init__(self, k=5, lambda_=3.0):
        self.k = k
        self.lambda_ = lambda_
    
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
        """Calculate local outlier probability for each sample in X

        Note: the local outlier probability is undefined for duplicated points
        Duplicated points will be given assigned a value of nan.
        
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
        
        num_rows = X.shape[0]
        
        prob_dist = np.sqrt((distances**2).mean(axis=1))
        
        plof = np.empty(num_rows)
        for i in range(num_rows):
            plof[i] = prob_dist[i] / np.mean(prob_dist[indices[i]])
            plof[i] -= 1.0
        plof[np.isinf(plof)] = np.nan
        
        # nplof : the std of plof assuming mean is zero
        nplof = self.lambda_ * np.sqrt(np.nanmean(plof**2))
        
        loop = erf(plof / nplof / np.sqrt(2)).clip(0)

        return loop
