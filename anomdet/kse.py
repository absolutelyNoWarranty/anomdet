# -*- coding: utf-8 -*-

from .base import BaseAnomalyDetector

import numpy as np

from sklearn.utils import check_random_state
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist

class KSE(BaseAnomalyDetector):
    """KSE - Average Kolmogorov-Smirnov Statistic for Outlier Detection
    References
    ----------

    .. [1] Kim, Michael S, "Robust, Scalable Anomaly Detection for Large Collections of Images", SocialCom, 2013.

    .. [2] MATLAB implementation by author : http://fr.mathworks.com/matlabcentral/fileexchange/39593-anomaly-detection/content/kse_test_matlab/kse_test.m
    """
    
    def __init__(self, subsample_size=0.25, random_state=None):
        self.random_state = random_state
        self.subsample_size = subsample_size
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        rs = check_random_state(self.random_state)

        nrows, dcols = X.shape
        scores = np.zeros(nrows)

        if self.subsample_size <= 1.0:
            nsample = int(np.ceil(self.subsample_size * nrows))
        else:
            nsample = self.subsample_size
            
        for i in range(nrows):
            # sample points to build dpop
            tmp1 = rs.choice(nrows, nsample, replace=False)
            dpop = X[tmp1, :]
            
            #dist_sample0 = np.zeros(nsample)
            #for j in range(nsample): # build distances from point i to sampled points
            #    dist_sample0[j] = np.sqrt(np.sum((dpop[j,:] - X[i,:])**2))
            dist_sample0 = cdist(X[i:i+1, :], dpop).flatten()
            
            
            tmp2 = rs.choice(nrows, nsample, replace=False)
            bpop = X[tmp2, :]
            
            #for k in range(nsample):
            #    dist_sample_temp = np.zeros(nsample)
            #    for j in range(nsample):
            #        dist_sample_temp[j] = np.sqrt(np.sum((dpop[j,:] - bpop[k,:])**2)) 
            #        
            #
            #    ks2stat, _ = ks_2samp(dist_sample0, dist_sample_temp)
            #    scores[i] += ks2stat / nsample
                
            dist_sample_temp = cdist(bpop, dpop)
            for row in dist_sample_temp:
                ks2stat, _ = ks_2samp(dist_sample0, row)
                scores[i] += ks2stat
            scores[i] /= nsample
        return scores
        
        
def kse_test(X, nsample=0.95, random_state=None):
    # Compute the outlier score for each p-dimensional data point
    # The highest scores are possible outliers, scores between [0,1]
    # Original scoring algorithm by Michael S Kim (mikeskim@gmail.com)
    # Version 1.00 (12/22/2012) for Matlab ported from R
    # not fully tested on Matlab, tested on GNU Octave and R
    rs = check_random_state(random_state)

    nrows, dcols = X.shape
    scores = np.zeros(nrows)

    if nrows <= 300:
        nsample = nrows
        #nsample = 1.0
    else:
        nsample = int(np.ceil(nsample * nrows))

    if nsample > 300:
        nsample = 300
        
    for i in range(nrows):
        # sample points to build dpop
        tmp1 = rs.choice(nrows, nsample, replace=False)
        dpop = X[tmp1, :]
        
        #dist_sample0 = np.zeros(nsample)
        #for j in range(nsample): # build distances from point i to sampled points
        #    dist_sample0[j] = np.sqrt(np.sum((dpop[j,:] - X[i,:])**2))
        dist_sample0 = cdist(X[i:i+1, :], dpop).flatten()
        
        
        tmp2 = rs.choice(nrows, nsample, replace=False)
        bpop = X[tmp2, :]
        
        #for k in range(nsample):
        #    dist_sample_temp = np.zeros(nsample)
        #    for j in range(nsample):
        #        dist_sample_temp[j] = np.sqrt(np.sum((dpop[j,:] - bpop[k,:])**2)) 
        #        
        #
        #    ks2stat, _ = ks_2samp(dist_sample0, dist_sample_temp)
        #    scores[i] += ks2stat / nsample
            
        dist_sample_temp = cdist(bpop, dpop)
        for row in dist_sample_temp:
            ks2stat, _ = ks_2samp(dist_sample0, row)
            scores[i] += ks2stat
        scores[i] /= nsample
    return scores