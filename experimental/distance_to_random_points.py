# -*- coding: utf-8 -*-

from ..base import BaseAnomalyDetector

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from ..datasets.digits import get_subsample_indices


class DistanceToRandomPoints(BaseAnomalyDetector):
    """DistanceToRandomPoints
    Randomly sample points from X and calculate distances to get an outlier score.
    """
    
    def __init__(self, subsample_size=0.25, random_state=None):
        self.subsample_size = subsample_size
        self.random_state = random_state
        
    def fit(self, X=None, y=None):
        return self
          
    def predict(self, X):
        random_state = check_random_state(self.random_state)
        (n, m) = X.shape
        n_to_sample = int(np.ceil(self.subsample_size * n))
        scores = np.zeros(n)
        for i in range(n):
            ind = random_state.choice(n, n_to_sample, replace=True)    
            scores[i] = np.mean(cdist(X[i:i+1, :], X[ind, :]))
        return scores
