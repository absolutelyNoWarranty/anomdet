# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA

def pca_reconstruction_error(X, k, sum_top=False):
    '''
    The sum of the squares of the standarized principal component scores of the unused components.
    
    X : data
    k : number of eigvals to use for reconstruction
        So (total - k) is the number of pca scores which will be summed
    sum_top : bool (default: False)
        By default, take the sum of the bottom scores, which is equivalent to calculating the reconstruction using the top k principle components.
        If true, will sum the top k scores instead.
    '''
    
    pca = PCA().fit(X)
    
    X_proj = pca.transform(X)
    if not sum_top:
        sum_of_scores = np.sum(np.sqrt(X_proj[:,k:]**2 / pca.explained_variance_[k:]), axis=1)
    else:
        sum_of_scores = np.sum(np.sqrt(X_proj[:,:k]**2 / pca.explained_variance_[:k]), axis=1)
    return sum_of_scores