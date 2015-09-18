# -*- coding: utf-8 -*-

from ..base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.special import erf

class LoOP(BaseAnomalyDetector):
    """LoOP : Local Outlier Probabilites

    Parameters
    ----------

    k : int
        Number of nearest neighbors to use for the nearest-neighbor query

    References
    ----------
    Kriegel, Hans-Peter, et al. "LoOP: local outlier probabilities." Proceedings of the 18th ACM conference on Information and knowledge management. ACM, 2009.
    """
    
    def __init__(self, k, lambda_=3.0):
        self.k = k
        self.lambda_ = lambda_
    
    def predict(self, A):
        """Calculate local outlier probability for each sample in A

        Parameters
        ----------
        A : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        lof : array, shape (n_samples,)
            Local outlier factor for each sample.
        """
        nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(A)
        distances, indices = nbrs.kneighbors(A)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        
        num_rows = A.shape[0]
        
        prob_dist = np.sqrt((distances**2).mean(axis=1))
        
        plof = np.empty(num_rows)
        nplof = 0.  # the std of plof assuming mean is zero
        for i in range(num_rows):
            plof[i] = prob_dist[i] / np.mean(prob_dist[indices[i]])
            plof[i] -= 1.0
            nplof += plof[i]**2
        nplof = self.lambda_ * np.sqrt(nplof / num_rows)
        
        loop = erf(plof / nplof / np.sqrt(2)).clip(0)

        return loop
    
    def fit(self, A=None, y=None):
        if self.k <= 0 or not isinstance(self.k, int):
            raise ValueError("k needs to be a positive integer.")
        self.A_ = A
        return self