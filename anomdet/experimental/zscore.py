# -*- coding: utf-8 -*-

from ..base import BaseAnomalyDetector

import warnings

import numpy as np
from scipy import stats
from sklearn.utils import check_random_state


class MultipleZScores(BaseAnomalyDetector):
    """MultipleZScores
    
    Calculate z-scores for each dimension
    
    Parameters
    ----------
    distribution : {'normal'}, optional (default='normal')
        The statistical distribution to assume.
        
    """
    
    def __init__(self, distribution="normal"):
        self.distribution = distribution
        
    def fit(self, X, y=None):
        '''
        Calculate empirical mean and std
        
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_dimensions)
        
        '''
        
        self.emp_mean = X.mean(axis=0)
        self.emp_std = X.std(axis=0)
        return self
        
    def predict(self, X):
        if not hasattr(self, 'emp_mean'):
            warnings.warn("Not fitted. Using X to fit empirical mean and "
                          "standard deviation.")
            self.fit(X)
        if self.distribution == "normal":
            zscores = np.abs((X - self.emp_mean) / self.emp_std)
            
            # ignore columns which has 0 std
            zscores = zscores[:, self.emp_std > 0]
            
            scores = np.sum(np.log(1. - stats.norm.cdf(zscores)), axis=1)
        
        return scores
        