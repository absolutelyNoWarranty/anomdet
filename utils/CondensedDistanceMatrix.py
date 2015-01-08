import numpy as np

from scipy.spatial.distance import pdist

def pdist_wrapper(X, metric='euclidean', p=2, w=None, V=None, VI=None):
    return CondensedDistanceMatrix(pdist(X, metric, p, w, V, VI))

class CondensedDistanceMatrix(object):
    """
    Wrapper for a condensed distance matrix (i.e. the kind returned by pdist)
    """
    
    def __init__(self, X):
        assert len(X.shape) == 1
        self.X = X
        self.n_samples = int((1 + np.sqrt(1 + 8*len(X))) / 2)
        assert (self.n_samples) * (self.n_samples-1) / 2 == len(X)
        
    def __getitem__(self, tup):
        i, j = tup
        if i < 0 or j < 0:
            raise IndexError("No support for negative indices yet!")
        
        if i >= self.n_samples:
            raise IndexError("index %d is out of bounds for axis 0 with size %d" % (i, self.n_samples))
        if j >= self.n_samples:
            raise IndexError("index %d is out of bounds for axis 1 with size %d" % (i, self.n_samples))
        
        # Distance to self is assumed to be zero
        if i==j:
            return 0.
        
        if i > j:
           return self.__getitem__((j, i))
        
        # The condensed distance matrix is the upper-triangular half of the distance matrix
        # without the diagonal.
        # Example: for a 4x4 distance matrix the condensed matrix is a array of length 10
        # whose indices correspond to the distance matrix as shown
        # 0 1 2 3 4 
        #0  0 1 2 3
        #1    4 5 6
        #2      7 8
        #3        9
        #4
        
        idx = (self.n_samples - 1 + self.n_samples - 1 - i + 1)*i/2 + (j - i - 1)
        return self.X[idx]
        
        