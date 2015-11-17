# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA

from .base import BaseAnomalyDetector

def pca_reconstruction_error(X, k, sum_top=False):
    '''
    The sum of the squares of the standarized principal component scores of the unused components.
    
    X : data
    k : number of eigvals to use for reconstruction
        So (total - k) is the number of pca scores which will be summed
    sum_top : bool (default: False)
        By default, take the sum of the bottom scores, which is equivalent to calculating the reconstruction using the top k principal components.
        If true, will sum the top k scores instead.
    '''
    
    pca = PCA().fit(X)
    
    X_proj = pca.transform(X)
    if not sum_top:
        sum_of_scores = np.sum(np.sqrt(X_proj[:,k:]**2 / pca.explained_variance_[k:]), axis=1)
    else:
        sum_of_scores = np.sum(np.sqrt(X_proj[:,:k]**2 / pca.explained_variance_[:k]), axis=1)
    return sum_of_scores
    
class PrincipalComponentReconstructionError(BaseAnomalyDetector):
    '''
    Estimating outlier-ness as the error made when reconstruction the data using
    only the top (or bottom) k principal components.
    
    This is calculated by taking the sum of the projection onto the unused
    components.
    
    Parameters
    ----------
    n_components : int or float, optional (default=2)
        
        The number of components to use for reconstruction
        Then, assuming sum_top=False, (total_n_components - n_components) is the
        number of PCA projection scores which will be summed.
        
        If 
            int : 1 <= n_components < n_dimensions
                The exact number of components
            float : Then 0.0 < n_components < 1.0
                Is the fraction of the total number of principal components to
                use.
        
    sum_top : bool, optional (default=False)
        By default, take the sum of the bottom projection scores,
        which is equivalent to calculating the reconstruction using the top-k
        principal components.
        If True, will sum the top projection scores instead.
    '''
    
    def __init__(self, n_components, sum_top=False):
        self.n_components = n_components
        self.sum_top = sum_top
        
    def fit(self, X, y=None):
        
        if self.n_components < 1.0:
            k = int(np.round(X.shape[1]))
        else:
            k = self.n_components
        
        pca = PCA().fit(X)
        self.pca = pca
        self.k = k
        return self
    
    def predict(self, X):
        pca = self.pca
        X_proj = pca.transform(X)
        k = self.k
        
        if not self.sum_top:
            sum_of_scores = np.sum(np.sqrt(X_proj[:,k:]**2 / pca.explained_variance_[k:]), axis=1)
        else:
            sum_of_scores = np.sum(np.sqrt(X_proj[:,:k]**2 / pca.explained_variance_[:k]), axis=1)
            
        return sum_of_scores         