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
    
    def __init__(self, n_min=20, n_max=100, alpha=0.5):
        self.n_min = n_min
        self.n_max = 100
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
        # Simple error checking
        assert self.n_max > self.n_min
        
        nbrs = NearestNeighbors(n_neighbors=self.n_max).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        num_rows = X.shape[0]
        
        k_sigma = []
        for i in xrange(num_rows):
            # start at self.n_min
            n_ = self.n_min+1 # plus 1 to avoid self
            max_zscore_mdef = -np.inf
            for r in distances[i, n_:]:
                n = len(nbrs.radius_neighbors(X[i], self.alpha*r, return_distance=False)[0])
                #n_hat = 0.
                #for ind in indices[n_]:
                #    n_hat += len(nbrs.radius_neighbors(X[i], self.alpha*r))
                #n_hat /= len(indices[n_])
                
                neighborhood_n = [len(nbrs.radius_neighbors(X[ind], self.alpha*r, return_distance=False)[0]) for ind in indices[n_]]
                
                n_hat = np.mean(neighborhood_n)
                sigma = np.std(neighborhood_n)
                
                mdef = (n_hat - n) / n_hat
                sigma_mdef = sigma/n_hat
                
                ##mdef.append(mdef_i)
                ##sigma_mdef_i.append(sigma_mdef_i)
                
                zscore_mdef = mdef / sigma_mdef #k_sigma in paper
                #import pdb;pdb.set_trace()
                if max_zscore_mdef < zscore_mdef:
                    max_zscore_mdef = zscore_mdef
                    

            k_sigma.append(max_zscore_mdef)
            
        return k_sigma