import os
import numpy as np

_digits_X, _digits_y = None, None

def load_digits_kaggle(filepath=None, cache=True):
    global _digits_X
    global _digits_y
    
    if _digits_X is not None:
        return _digits_X, _digits_y
    
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'train.csv')
    
    X = np.loadtxt(filepath, delimiter=',', skiprows=1)
    y = X[:, 0].astype(int)
    X = X[:, 1:]
    
    if cache:
        _digits_X = X
        _digits_y = y
    return (X, y)

def get_subsample_indices(classes, y, select, replace=False):
    '''
    Returns indices which can be used to subsample a dataset X.
    Input:
        `y` - the class labels of X
        `select` - the number to select from each class
        `replace` - whether to sample with replacement or without
            default: False
    '''
    result = []
    #classes = np.unique(y)
    for num_to_select, class_ in zip(select, classes):
        indices = np.where(y==class_)[0]
        result.append(np.random.choice(indices, num_to_select, replace=replace))
    result = np.concatenate(result)
    
    return result
        
    
def benchmark_digits1(model, metric, n_iter=25, num_train_samples=1000, num_test_samples=1000, anomaly_ratio=0.05):
    
    n_train_normal = int(num_train_samples * (1.0 - anomaly_ratio))
    n_train_anomaly = num_train_samples - n_train_normal
    
    n_test_normal = int(num_test_samples * (1.0 - anomaly_ratio))
    n_test_anomaly = num_test_samples - n_test_normal
    
    all_X, all_y = load_digits_kaggle()
    
    perf_scores = []
    for experiment_iter in range(n_iter):
        for i in range(10):
            normal_class = i
            
            #binary_y
            bin_y = all_y != normal_class; classes = [True, False]
            
            train_ind = get_subsample_indices(classes, bin_y, select=[n_train_anomaly, n_train_normal])
            test_ind = get_subsample_indices(classes, bin_y, select=[n_test_anomaly, n_test_normal])

            X_train = all_X[train_ind]
            X_test = all_X[test_ind]
            y_train = bin_y[train_ind]
            y_test = bin_y[test_ind]
        
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            perf_scores.append(metric(y_test, preds))
            
    return np.mean(perf_scores)
            
def benchmark_digits2(model, metric, n_iter=25, num_train_samples=1000, num_test_samples=1000, anomaly_ratio=0.05):
    
    n_train_normal = int(num_train_samples * (1.0 - anomaly_ratio))
    n_train_anomaly = num_train_samples - n_train_normal
    
    n_test_normal = int(num_test_samples * (1.0 - anomaly_ratio))
    n_test_anomaly = num_test_samples - n_test_normal
    
    all_X, all_y = load_digits_kaggle()
    
    perf_scores = []
    for experiment_iter in range(n_iter):
        for i in range(10):
            anomaly_class = i
            
            #binary y
            bin_y = all_y == anomaly_class; classes = [True, False]
            
            train_ind = get_subsample_indices(classes, bin_y, select=[n_train_anomaly, n_train_normal])
            test_ind = get_subsample_indices(classes, bin_y, select=[n_test_anomaly, n_test_normal])

            X_train = all_X[train_ind]
            X_test = all_X[test_ind]
            y_train = bin_y[train_ind]
            y_test = bin_y[test_ind]
        
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            perf_scores.append(metric(y_test, preds))
            
    return np.mean(perf_scores)
    
# cached
_digits_X, _digits_y = load_digits_kaggle()