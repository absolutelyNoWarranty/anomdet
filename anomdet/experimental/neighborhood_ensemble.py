# -*- coding: utf-8 -*-

from ..base import BaseAnomalyDetector

import numpy as np
from scipy.spatial.distance import cdist

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from anomdet import LOF
from anomdet.lof.loop import LoOP
from scipy.special import erf


class NeighborhoodEnsemble(BaseAnomalyDetector):
    """Ensemble of Metrics calculated from Random Neighborhoods
    """
    
    def __init__(self, sample_size=255, n_samples=10, strategy="LoOP", random_state=None):
        self.sample_size = sample_size
        self.n_samples = n_samples
        self.strategy = strategy
        self.random_state = random_state
        
    def fit(self, X=None, y=None):
        return self
        
    def predict(self, X):
        rs = check_random_state(self.random_state)
        n_neighbors = 10
        lambda_ = 3. 
        n_items = X.shape[0]
        
        # samples : a list of random samples of X (where each has shape (sample_size, X.shape[1]))
        samples = []
        for k in range(self.n_samples):
            ind = rs.choice(n_items, self.sample_size, replace=False)
            samples.append(X[ind, :])
         
        outlier_scores = np.zeros(X.shape[0])
        for k in range(self.n_samples):
            nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(samples[k])
            distances, indices = nbrs.kneighbors(samples[k])
            indices = indices[:, 1:]
            distances = distances[:, 1:]
            
            prob_dist = np.sqrt((distances**2).mean(axis=1))
            
            plof = np.empty(samples[k].shape[0])
            for i in range(samples[k].shape[0]):
                plof[i] = prob_dist[i] / np.mean(prob_dist[indices[i]])
                plof[i] -= 1.0
            plof[np.isinf(plof)] = np.nan
            
            # nplof : the std of plof assuming mean is zero
            nplof = lambda_ * np.sqrt(np.nanmean(plof**2))
            
            for i in range(n_items):
                distances, indices = nbrs.kneighbors(X[i])
                plof_ = np.sqrt((distances**2).mean()) / np.mean(prob_dist[indices]) - 1.0
                
                outlier_scores[i] += erf(plof_ / nplof / np.sqrt(2))  #.clip(0), don't clip to 0 like in LoOP
        
        self.samples = samples
        outlier_scores /= self.n_samples
        return outlier_scores
        
class LoOP_Random(BaseAnomalyDetector):
    """LoOP w/ random neighborhoods : Local Outlier Probabilites

    Parameters
    ----------
    k : int
        Number of nearest neighbors to use for the nearest-neighbor query

    References
    ----------
    Kriegel, Hans-Peter, et al. "LoOP: local outlier probabilities." Proceedings of the 18th ACM conference on Information and knowledge management. ACM, 2009.
    """
    
    def __init__(self, k, lambda_=3.0, n_iter=10, random_state=None):
        self.k = k
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X=None, y=None):
        if self.k <= 0 or not isinstance(self.k, int):
            raise ValueError("k needs to be a positive integer.")
        self.X_ = X
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
        num_rows = X.shape[0]
        rs = check_random_state(self.random_state)
        
        outlier_scores = np.empty(num_rows)
        for round_i in range(self.n_iter):
            indices = rs.choice(num_rows, (num_rows, self.k*10))
            distances = np.empty((num_rows, self.k*10))
            for j in range(self.k*10):
                distances[:, j] = np.sqrt(((X - X[indices[:, j]])**2).sum(axis=1))
            
            # Out of the 3*k random neighborhoods get the k-nearest neighbors
            sort_ind = np.argsort(distances, axis=1)[:, :self.k]
            indices = indices[np.arange(num_rows).reshape(num_rows, 1), sort_ind]
            distances = distances[np.arange(num_rows).reshape(num_rows, 1), sort_ind]

            
            prob_dist = np.sqrt((distances**2).mean(axis=1))
            
            plof = np.empty(num_rows)
            for i in range(num_rows):
                plof[i] = prob_dist[i] / np.mean(prob_dist[indices[i]])
                plof[i] -= 1.0
            plof[np.isinf(plof)] = np.nan
            
            # nplof : the std of plof assuming mean is zero
            nplof = self.lambda_ * np.sqrt(np.nanmean(plof**2))
            
            loop = erf(plof / nplof / np.sqrt(2)).clip(0)
            outlier_scores += loop
        outlier_scores /= self.n_iter
        return outlier_scores