# -*- coding: utf-8 -*-

from ..base import BaseAnomalyDetector

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.utils import check_random_state

from ..datasets.digits import get_subsample_indices


class DistanceToRandomPoints(BaseAnomalyDetector):
    """Distance To Random Points
    
    Randomly sample points from X and calculate distances to get an outlier score.
    
    This method is useful as a simple baseline to compare with other methods
    but it can also be used for real problems.
    
    Parameters
    ----------
    `subsample_size' : float, optional (default=0.25)
        Float between 0 and 1. How large the random sample of points should be
        relative to the total size of the input data.
    
    `strategy' : str, optional
        Strategy to use for getting the random neighborhood of points.
            * "sample_every_iteration" (default) : For every point, resample
              points from the dataset.
            * "sample_once" : Get one random sample and do all distance
              calculations with it.
    
    `random_state': int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.

    """
    
    def __init__(self, subsample_size=0.25, strategy="sample_every_iteration", random_state=None):
        self.subsample_size = subsample_size
        self.strategy = strategy
        self.random_state = random_state
        
    def fit(self, X=None, y=None):
        return self
          
    def predict(self, X):
        random_state = check_random_state(self.random_state)
        (n, m) = X.shape
        n_to_sample = int(np.ceil(self.subsample_size * n))
        scores = np.zeros(n)
        
        if self.strategy == "sample_every_iteration":
            for i in range(n):
                ind = random_state.choice(n, n_to_sample, replace=True)    
                scores[i] = np.mean(cdist(X[i:i+1, :], X[ind, :]))
                
        elif self.strategy == "sample_once":
            ind = random_state.choice(n, n_to_sample, replace=True)
            scores = np.mean(cdist(X, X[ind, :]), axis=1)
            
        else:
            raise ValueError("Unknown strategy type.")
        
        return scores
