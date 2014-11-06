from .base import BaseAnomalyDetector
import numpy as np

def power_method(A, x, tol, maxiter):
    '''
    Compute the eigenvector via power method
    
    Input:
        A : the data matrix, each row represents an instance
        x : initial vector
        tol : the convergence tolerance
        maxiter : the maximum number of iterations
    
    Output:
        lambda_ : the resulting eigenvalue
        v : the resulting eigenvector
    '''
    relerr = np.inf
    niter = 1
    
    while relerr >= tol and niter < maxiter:
        z = x/np.linalg.norm(x, 2)
        x = A.dot(z)
        alpha1 = z.T.dot(x)
        if niter > 1:
            relerr = np.abs(alpha1-alpha0)/np.abs(alpha0)
        alpha0 = alpha1
        niter += 1
        
    lambda_ = alpha1
    v = z
    return (lambda_, v)

class OversamplingPCA(BaseAnomalyDetector):
    """Outlier Detection via Over-sampling PCA
    Using the variation of the first principal direction detect the
    outlier-ness of each instance(event) in the leave one out procedure.
    Here the over-sampling on target instance is also used for enlarge the 
    outlierness
    """
    
    def __init__(self, oversampling_ratio):
        self.oversampling_ratio = oversampling_ratio
    
    def predict(self, A):
        """Calculate local outlier factor for each sample in A

        Parameters
        ----------
        A : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        suspicious_score : The suspicious score for each instance
        """
        tol = 10e-20
        maxiter = 500
        
        ratio = self.oversampling_ratio
        
        (n, p) = A.shape
        A_m = A.mean(axis=0)
        out_prod = A.T.dot(A)/n  # outer product
        _, u = power_method(out_prod - A_m[:,None]*A_m, np.ones((p, 1)), tol, maxiter)

        # Start the Leave-One-Out procedure with over-sampling PCA\
        sim_pool = np.zeros((n, 1))
        for i in range(n):
            temp_mu = (A_m + ratio*A[i, :]) / (1 + ratio)  # update mean
            temp_cov = (out_prod + ratio*A[i, :, None] * A[i, :]) / (1 + ratio) - temp_mu[:,None] * temp_mu
            _, u_temp = power_method(temp_cov, u, tol, maxiter)
            sim_pool[i, :] = np.abs(np.diag(u.T.dot(u_temp)))  # compute cosine similarity
        
        return 1. - sim_pool
    
    def fit(self, A=None, y=None):
        self.A_ = A
        return self