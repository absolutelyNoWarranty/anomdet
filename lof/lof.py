from ..base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np

class LOF(BaseAnomalyDetector):
    '''
    Local Outlier Factor
    '''
    
    def __init__(self, k):
        self.k = k
    
    def predict(self, A):
        '''
        Predict anomaly scores for dataset X
        '''
        nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(A)
        distances, indices = nbrs.kneighbors(A)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        
        k_dists = distances[:, -1]
        
        num_rows = A.shape[0]
        
        lrd_value = np.zeros((num_rows, 1));
        lrd_value = np.zeros(num_rows)
        for i in xrange(num_rows):
            temp = A[i, :] - A[indices[i], :]
            temp = np.sqrt(np.sum(temp**2, 1))
            reachability_dists = np.max(np.vstack([temp, k_dists[indices[i]]]), 0)
            lrd_value[i] = self.k/sum(reachability_dists);
        
        lof = np.zeros(num_rows)
        for i in xrange(num_rows):
            lof[i] = np.sum(lrd_value[indices[i]]) / lrd_value[i] / self.k
        
        return(lof)
    
    def fit(self, A=None, y=None):
        if self.k <= 0 or not isinstance(self.k, int):
            raise ValueError("k needs to be a positive integer.")
        self.A_ = A
        return self
    
    #def lrd(A, index_p, k_dist, k_index, numneighbors):
    #    temp = A[index_p, :] - A[k_index[]]