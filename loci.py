# -*- coding: utf-8 -*-

from .base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np

class LOCI(BaseAnomalyDetector):
    """The LOCI method

    Parameters
    ----------

    n_min : int
        Number of neighbors that must be included in the smallest radius
        (default: 20)
        
    alpha : float
        Ratio of counting neighborhood radius to sampling neighborhood radius
        (default: 0.5)
    

    References
    ----------
    S. Papadimitriou, H. Kitagawa, P. B. Gibbons "LOCI: Fast Outlier Detection Using the Local Correlation Integral"
    """
    
    def __init__(self, n_max=100, alpha=0.5):
        self.n_max = n_max
        self.alpha = alpha
    
    def fit(self, X=None, y=None):
        pass
    
    def predict(self, X):
        """Calculate MDEF vs sigma_mdef for each sample in X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        lof : array, shape (n_samples,)
            Local outlier factor for each sample.
        """
        
        nbrs = NearestNeighbors(n_neighbors=self.n_max).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        num_rows = X.shape[0]
        
        k_sigma = []
        for i in xrange(num_rows):
            max_zscore_mdef = -np.inf
            for j in range(0, distances.shape[1]):
                r = distances[i, j]
                
                n = np.sum(distances[i] <=  self.alpha*r)
                # note that indices[i] includes i
                # indices[i, :(j+1)] => j+1 neighbors of point i (point i plus additional j neighbors)
                # distances[...] the neighbor distances for point i and point i's j neighbors
                # neighborhood_n => counts for the neighborhood size of each of the points in point i's neighborhood
                neighborhood_n = (distances[indices[i, :(j+1)]] <= self.alpha*r).sum(axis=1)
                n = neighborhood_n[0]
                n_hat = np.mean(neighborhood_n)
                sigma = np.std(neighborhood_n)
                
                mdef = (n_hat - n) / n_hat
                sigma_mdef = sigma/n_hat
                          
                #k_sigma in paper
                if mdef < 1e-16:
                    zscore_mdef = 0.0
                else:
                    zscore_mdef = mdef / sigma_mdef

                if max_zscore_mdef < zscore_mdef:
                    max_zscore_mdef = zscore_mdef
                    

            k_sigma.append(max_zscore_mdef)
            
        return k_sigma