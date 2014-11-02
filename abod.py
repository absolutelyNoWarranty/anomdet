from .base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import norm

class ABOD(BaseAnomalyDetector):
    '''
    Angle Based Outlier Detector
    
    Uses the fast ABOD algorithm of "Angle-Based Outlier Detection in High-dimensional Data In KDD2008"
    [Hans-Peter, Kriegel Matthias, Schubert Arthur Zimek]
    '''
    
    def __init__(self, n_k):
        self.n_k = n_k
    
    def predict(self, A):
        '''
        Predict anomaly scores for dataset X
        '''
        #import pdb;pdb.set_trace()
        num_instances = A.shape[0]
        var_array = []
        n_k = min(self.n_k, num_instances)
        
        for i in range(num_instances):
            var_front = 0
            var_back = 0
            denominator = 0
            Temp = A[i, :] - A
            Temp = np.sum(Temp**2, axis=1)
            index = np.argsort(Temp)
            index = index[1:n_k]
            #index = index.T
            count = 0
            for j in index:
                count += 1
                for k in index[count:]:
                    vector1 = A[j, :] - A[i, :]
                    vector2 = A[k, :] - A[i, :]
                    norm_vector1Xnorm_vector2 = norm(vector1) * norm(vector2)
                    vector1Xvector2T = vector1.dot(vector2)
                    var_front = var_front + (1. / norm_vector1Xnorm_vector2) * (vector1Xvector2T / (norm_vector1Xnorm_vector2**2)) **2
                    var_back = var_back + (vector1Xvector2T / norm_vector1Xnorm_vector2**3)
                    denominator += 1./norm_vector1Xnorm_vector2
            var_array.append(var_front/denominator - (var_back/denominator)**2)
        
        min_var_array = min(var_array)
        abof = (np.array(var_array) - min_var_array) / (max(var_array) - min_var_array)
        return abof
        
    def fit(self, A=None, y=None):
        self.A_ = A
        return self