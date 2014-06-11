# -*- coding: utf-8 -*-

from ..base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np

class LOF(BaseAnomalyDetector):
    """Calculate Local Outlier Factor of data

    Parameters
    ----------

    k : int
        Number of nearest neighbors to use

    Returns
    -------

    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers

    labels : array, shape (n_samples,)
        cluster labels for each point

    References
    ----------
    Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, and Jörg Sander. 2000. LOF: identifying density-based local outliers. SIGMOD Rec. 29, 2 (May 2000), 93-104. DOI=10.1145/335191.335388 http://doi.acm.org/10.1145/335191.335388
    """
    
    def __init__(self, k):
        self.k = k
    
    def predict(self, A):
        """Calculate local outlier factor for each sample in A

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