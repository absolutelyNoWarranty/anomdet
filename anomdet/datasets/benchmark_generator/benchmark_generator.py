import numpy as np
import pandas as pd
import load_datasets as ld
import os

from sklearn.datasets.base import Bunch
from sklearn.metrics import auc_score as roc_auc_score
from misc import precision_at_k

def n_choose_k(n, k):
    '''
    Randomly return k numbers from the range [0,n)
    '''
    return np.random.permutation(n)[:k]

def n_choose_k1k2(n, k1, k2):
    '''
    Randomly return k1, k2 numbers from the range [0,n)
    Used to split train, test
    OR if n is np.array of logicals TRue, False
    returns the k1,k2 number of indices only where
    is True
    '''
    if isinstance(n, int):
        perm = np.random.permutation(n)
        return (perm[:k1], perm[k1:k1+k2])
    elif isinstance(n, np.ndarray) and n.dtype == 'bool':
        indices = np.where(n)[0]
        len_ = len(indices)
        assert k1+k2 <= len_
        perm = np.random.permutation(len_)
        return (indices[perm[:k1]], indices[perm[k1:k1+k2]])
# gas sensor (1000
# landsat satellite  (750
# shuttle (1000
    
    
def iter_views(dataset, num_replicates=10, min_normal_class_size = 1000, train_anomaly_ratio=0.10, test_anomaly_ratio = 0.01):
    bunch = dataset()
    
    n_data, n_dim = bunch.data.shape
    classes = pd.Series(bunch.target, dtype='int').value_counts().to_dict()
    
    # classes with at least `min_normal_class_size` instances
    normal_classes = [(c, n_ins) for c, n_ins in classes.items() if n_ins >= 2*min_normal_class_size]
    
    # get training data set
    for normal_class, size in normal_classes:
        normal_class_size = int(size*0.5) # split equally between train and test
        anomaly_class_size_train = np.rint(normal_class_size * train_anomaly_ratio)
        anomaly_class_size_test = np.rint(normal_class_size * test_anomaly_ratio)
        anomaly_classes = []
        for C, n_ins in classes.items():
            if C == normal_class: continue
            if n_ins < anomaly_class_size_train + anomaly_class_size_test: continue
            
            anomaly_classes.append(C)
        
        num_anom_types = len(anomaly_classes)
        
        if num_anom_types == 0:
            print "No anomaly classes available for normal class=%d"  % normal_class
            continue
        
        
        
        # Now subsample and yield `num_replicate` datasets
        
        for i in range(num_replicates):
            train_ind = []
            test_ind = []
        
            train_normal_ind, test_normal_ind = n_choose_k1k2(bunch.target==normal_class, normal_class_size, normal_class_size)
            train_ind.append(train_normal_ind)
            test_ind.append(test_normal_ind)
            
            for anom_class in anomaly_classes:
                train_anom_ind, test_anom_ind = n_choose_k1k2(bunch.target==anom_class, anomaly_class_size_train, anomaly_class_size_test)
                
                train_ind.append(train_anom_ind)
                test_ind.append(test_anom_ind)
            train_ind = np.concatenate(train_ind)
            test_ind = np.concatenate(test_ind)
            
            yield Bunch(train_data=bunch.data[train_ind],
                        test_data=bunch.data[test_ind],
                        train_labels=np.repeat(np.array(['normal']+['anomaly'+str(j) for j in range(1, num_anom_types+1)]), [normal_class_size]+[anomaly_class_size_train]*num_anom_types),
                        test_labels=np.repeat(np.array(['normal']+['anomaly'+str(j) for j in range(1, num_anom_types+1)]), [normal_class_size]+[anomaly_class_size_test]*num_anom_types),
                        normal_class_=normal_class,
                        anomaly_classes_=anomaly_classes,
                        original_ind_train_=train_ind,
                        original_ind_test_=test_ind)

class IDoNothing():
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        pass
        return self
    def predict(self, X):
        '''
        `k` - how many anomalies to return
        '''
        
        n_data = X.shape[0]
        return np.random.rand(n_data)
                        
def run_benchmark(ADAs, dataset):
    aucs = []
    patks = [] #precision at ks 
    for dataset_ in iter_views(dataset, num_replicates=1):
        aucs.append([])
        patks.append([])
        for ada in ADAs:
            train_data = dataset_.train_data.copy()
            test_data = dataset_.test_data.copy()
            ada.fit(train_data, (dataset_.train_labels!='normal').astype(int))
        
            preds = ada.predict(test_data)
        
            auc = roc_auc_score((dataset_.test_labels!='normal').astype(int), preds)
            p_at_k = precision_at_k((dataset_.test_labels!='normal').astype(int), preds)
        
            aucs[-1].append(auc)
            patks[-1].append(p_at_k)
    return (aucs, patks)

if __name__ == '__main__':
    #from ocsvm import OneNormalClassSVM
    #from multiple_anom_class_svm import MultipleAnomClassSVM
    results = run_benchmark([IDoNothing()], ld.load_gas_sensor_array_drift)