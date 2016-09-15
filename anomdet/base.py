from sklearn.base import BaseEstimator
from .datasets import OutlierDataset

class BaseAnomalyDetector(BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X=None, y=None):
        '''
        One-class algorithms need a fit stage
        '''
        return self
        
    def predict(self, X):
        '''
        Predict anomaly scores for dataset X
        '''
        pass
    
    def benchmark(self, dataset):
        '''
        Benchmark on `dataset`
        '''
        if not isinstance(dataset, OutlierDataset):
            raise ValueError("dataset is not an OutlierDataset")
        
        return dataset.benchmark(self)
        
    def select(self, threshold=None, top_n=None, percentage=None):
        '''
        Select suspicious instances from the data
        '''
        pass
        
    def auc_score(self):
        pass
        
    def score(self):
        pass
    
    