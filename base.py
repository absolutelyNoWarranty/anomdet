from sklearn.base import BaseEstimator

class BaseAnomalyDetector(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        '''
        One-class algorithms need a fit stage
        '''
        return self
        
    def predict(self, X):
        '''
        Predict anomaly scores for dataset X
        '''
        pass
    
        
    def select(self, threshold=None, top_n=None, percentage=None):
        '''
        Select suspicious instances from the data
        '''
        pass
    def auc_score(self):
        pass
    def score(self):
        pass
    
    