from ..base import BaseAnomalyDetector
from sklearn.mixture import GMM

class MixtureOfGaussians(BaseAnomalyDetector):
    '''
    Wrapper over sklearn's GMM class
    '''
    
    def __init__(self, n_clusters=1, covariance_type='diag', random_state=None, n_iter=100, n_init=1):
        #self.n_clusters = n_clusters
        #self.covariance_type = covariance_type
        #self.random_state = random_state
        #self.n_iter = n_iter
        #self.n_init = n_init
        self.gmm = GMM(n_components=n_clusters, covariance_type=covariance_type, random_state=random_state, n_iter=n_iter, n_init=n_init)
        
    def fit(self, X, y=None):
        self.gmm.fit(X)
        return self
        
    def predict(self, X):
        logprob = self.gmm.score_samples(X)[0]
        return -logprob