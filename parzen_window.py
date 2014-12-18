# -*- coding: utf-8 -*-

from .base import BaseAnomalyDetector

from scipy.stats import gaussian_kde

class ParzenWindow(BaseAnomalyDetector):
    """Rank points using Kernel Density estimates (parzen windows)
    Wrapper around scipy's gaussian_kde
    
    Parameters
    ----------

    bandwidth : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.

    """
    
    def __init__(self, bandwidth='silverman'):
        self.bandwidth = bandwidth
    
    def predict(self, X):
        return 1.0 - self.kde.evaluate(X.T)
    
    def fit(self, X, y=None):
        self.kde = gaussian_kde(X.T, bw_method=self.bandwidth)
        return self