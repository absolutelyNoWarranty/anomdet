from .base import BaseAnomalyDetector
import numpy as np

def power_method(A, x=None, tol=1e-19, maxiter=500, random_state=None):
    '''
    Compute the dominant eigenvector of A via the power iteration method
    
    Input:
        A : the data matrix, each row represents an instance
        x : initial vector, optional (default=None)
        tol : the convergence tolerance, optional (default=1e-19)
        maxiter : the maximum number of iterations, optional (default=500)
        random_state : int, or RandomState, optional (default=None)
            if not given will use seed=0
    
    Output:
        lambda_ : the resulting eigenvalue
        v : the resulting eigenvector
        
    Refererence:
    https://en.wikipedia.org/wiki/Power_iteration
    http://www.netlib.org/utk/people/JackDongarra/etemplates/node95.html
    '''
    if x is None:
        if random_state==None:
            rs = np.random.RandomState(0)
        x = rs.rand(A.shape[1])
        
    relerr = np.inf
    niter = 1
    
    while relerr >= tol and niter < maxiter:
        z = x/np.linalg.norm(x, 2)
        x = A.dot(z)
        alpha1 = z.dot(x)
        if niter > 1:
            relerr = np.abs(alpha1-alpha0)/np.abs(alpha0)
        alpha0 = alpha1
        niter += 1
        
    lambda_ = alpha1
    v = z
    return (lambda_, v)

class OversamplingPCA(BaseAnomalyDetector):
    """Outlier Detection via Over-sampling PCA
    Measure outlier-ness of data as the perturbation in the first principal
    component after oversampling.
    """
    
    def __init__(self, oversampling_ratio):
        self.oversampling_ratio = oversampling_ratio
    
    def predict(self, A):
        """Calculate outlier score for each sample in A

        Parameters
        ----------
        A : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        suspicious_score : The suspicious score for each instance
        """
        
        ratio = self.oversampling_ratio
        
        (n, _) = A.shape
        A_m = self._A_m
        out_prod = self._out_prod
        u = self._u

        # Start the Leave-One-Out procedure with over-sampling PCA
        similarities = np.zeros(n)
        for i in range(n):
            # Update the mean and covariance with weighted x_i
            temp_mu = (A_m + ratio*A[i, :]) / (1 + ratio)
            temp_cov = (out_prod + ratio*A[i, :, None] * A[i, :]) / (1 + ratio) - temp_mu[:,None] * temp_mu
            _, u_temp = power_method(temp_cov, x=u)
            
            # Compute absolute cosine similarity between eigenvector with
            #oversampled point and the original dominant eigenvector.
            similarities[i] = np.abs(u.T.dot(u_temp))  
        
        # Return cosine distances as outlier scores
        return 1. - similarities
    
    def fit(self, A):
        '''
        Fit OversamplingPCA. Find the dominant eigenvector of A.
        
        Parameters
        ----------
        A : array-like shape (n_samples, n_features)
        
        '''
        (n, _) = A.shape
        A_m = A.mean(axis=0)
        out_prod = A.T.dot(A)/n  # outer product
        
        # Find top eigenvector of covariance matrix.
        # outer product of A minus outer product of mean = empirical covariance
        _, u = power_method(out_prod - A_m[:,None] * A_m)
        
        self._A_m = A_m
        self._out_prod = out_prod
        self._u = u
        
        return self