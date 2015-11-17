# -*- coding: utf-8 -*-

from base import BaseAnomalyDetector

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

class RandomWalkOutlier(BaseAnomalyDetector):
    """Outlier Detection using Random Walks ("OutRank" a la "PageRank")
    
    damping_factor : float, damping factor, must be greater than or equal to 0 and less than 1
    similarity_function : str, the similarity measure to use
        'cosine' - Cosine Similarity
        'rbf' - RBF Similarity
    References
    ----------

    .. [1] Moonesinghe, H. D. K.; Tan, Pang-Ning, "Outlier Detection using Random Walks"
    
    """
    
    POSSIBLE_SIMILARITY_MEASURES = ['cosine', 'rbf']
    
    def __init__(self, damping_factor=0.1, similarity_measure='cosine'):
        self.damping_factor = damping_factor
        self.similarity_measure = similarity_measure
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        # create transition matrix A
        try:
            sim = RandomWalkOutlier.POSSIBLE_SIMILARITY_MEASURES.index(self.similarity_measure)
        except ValueError as err:
            print "similarity_measure should be one of: %s" % " ".join(RandomWalkOutlier.POSSIBLE_SIMILARITY_MEASURES)
            raise
        if sim == 0:
            S = cosine_similarity(X)
        elif sim == 1:
            S = rbf_kernel(X)
        S[range(S.shape[0]), range(S.shape[1])] = 0.
        # normalize rows and add in damping factor
        A = (S / S.sum(axis=1)[:,None])
        A = A * (1. - self.damping_factor) + (self.damping_factor / A.shape[1])
        
        # power method for finding eigenvector
        c = np.ones(X.shape[0]) / X.shape[0]
        diff = np.inf
        while diff > 1e-16:
            c_ = c
            c = A.T.dot(c)
            diff = np.linalg.norm(c_ - c, 1)
            print diff
            
        return 1 - c