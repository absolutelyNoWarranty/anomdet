# -*- coding: utf-8 -*-

from .base import BaseAnomalyDetector

from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import eigh

class KPCA(BaseAnomalyDetector):
    """Kernel PCA for novelty detection
    Uses the Gaussian Kernel.
    
    Parameters
    ----------
    
    sigma : float
            Width of Gaussian kernel
    n_eigval : int
               Number of eigenvalues to be extracted

    References
    ----------
    Heiko Hoffmann "Kernel PCA for novelty detection"
    http://www.heikohoffmann.de/kpca.html
    """
    
    def __init__(self, sigma, n_eigval):
        self.sigma = sigma
        self.n_eigval = n_eigval  # number of eigenvalues to use during pred.
    
    def predict(self, A):
        return self._reconstruction_error(A, self.n_eigval)
    
    def _gaussian_kernel(self, A, sigma):
        '''
        If A is of shape (n, d), where rows are data points
        Then the desired kernel matrix is a n x n matrix
        where (i, j)-th element is a function of A[i, :] - A[j, :]
        Specifically it is a function of the dot product of that difference
        which, when expanded, is
        A[i, :].dot(A[i, :].T) + A[j, :].dot(A[j, :].T) - 2*A[i, :].dot(A[j, :].T)
        '''
        n = A.shape[0]
        squares = np.sum(A**2, axis=1)  # i element is the sum over the i-th row of A (dot-product itself)
        squares = squares.reshape(n, 1) + squares.reshape(1, n)
        return np.exp(-(squares - 2*A.dot(A.T)) / (2*sigma*sigma))
    
    def _z_data_projection(self, z):
        '''
        Projects z using kernel function onto data
        '''
        n, d = self.data.shape
        z = z.reshape(1, d)
        sigma = self.sigma
        return np.exp(- (z.dot(z.T) - 2*self.data.dot(z.T) + np.sum(self.data**2, axis=1).reshape(n, 1)) /(2*sigma*sigma)).flatten()
    
    def _reconstruction_error(self, data, n_components):
        '''
        Returns the reconstruction error for KPCA
        '''
        n = data.shape[0]
        
        
        lambda_ = self._kernel_eigvals[:self.n_eigval]
        alpha = self._kernel_eigvecs[:, :self.n_eigval]
        alpha = alpha / np.sqrt(lambda_)
        
        #precompute helper vectors
        sumalpha = np.sum(alpha, 0)
        alphaKrow = self._mean_uncentered_K_row.dot(alpha)
        #Ksum = np.sum(self.K)
        #Kavg = np.sum(self.K)/n/n
        
        err = np.empty(n)
        for i in range(n):
            z = data[i, :]
            proj_on_data = self._z_data_projection(z)
            
            # Projections onto components
            f = proj_on_data.dot(alpha) - alphaKrow - np.sum(proj_on_data)/n * sumalpha + self._mean_uncentered_K * sumalpha

            # Spherical Potential
            s = self._gaussian_kernel(z[None,:], self.sigma) - 2.*np.sum(proj_on_data)/n + self._mean_uncentered_K

            err[i] = s - f.dot(f)
        
        return err
        
    def fit(self, data, y=None):
        n, d = data.shape
        
        # Kernel matrix
        K = self._gaussian_kernel(data, self.sigma)
    
        # Center K in feature space
        Krow = np.sum(K, axis=1) / n
        Ksum = np.sum(Krow) / n
        
        for i in range(n):
            for j in range(n):
                K[i, j] = K[i, j] - Krow[i] - Krow[j] + Ksum
        
        # Calculate sorted eigen-vals/vecs
        eigvals, eigvecs = eigh(K)
        ind = np.argsort(eigvals)[-1::-1]  # reverse to put largest first
        eigvals = eigvals[ind]
        eigvecs = eigvecs[:, ind]
        
        self._kernel_eigvals = eigvals
        self._kernel_eigvecs = eigvecs
        self.K = K
        self._mean_uncentered_K_row = Krow
        self._mean_uncentered_K = Ksum
        self.data = data
        return self
    