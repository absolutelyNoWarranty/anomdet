import os

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_random_state

def _benchmark(ada, dataset, random_state=None, use_transformed_features=False, predict_proba=False):
    assert dataset == 'letter' or dataset == 'speech'
    
    HERE = os.path.dirname(__file__)
    DATA_DIR = os.path.join(HERE, dataset)
    random_state = check_random_state(random_state)
    
    ORIG = dataset + '-orig.txt'
    LABELS = dataset + '-labels.txt'
    TRANS = dataset + '-trans.txt'
    
    X = np.loadtxt(os.path.join(DATA_DIR, ORIG), delimiter=',')
    y = np.loadtxt(os.path.join(DATA_DIR, LABELS), dtype=int)
    X_ft = np.loadtxt(os.path.join(DATA_DIR, TRANS), delimiter=',')
    
    aucs = []
    # paper used 20 iterations
    for i in range(20):
        
        # paper's train/test ratio: 60%/40%
        train_ind, test_ind = train_test_split(np.arange(X.shape[0]), test_size=0.4, random_state=random_state)
        X_train = X[train_ind]
        y_train = y[train_ind]
        X_test = X[test_ind]
        y_test = y[test_ind]

        X_train_ft = X_ft[train_ind]
        X_test_ft = X_ft[test_ind]


        # Change half of y_train's 1's to 0's to reflect a contaminated
        # normal set
        y_train_contaminated = y_train.copy()
        flip_ind = random_state.choice(sum(y_train), int(sum(y_train)*0.5), replace=False)
        y_train_contaminated[np.where(y_train)[0][flip_ind]] = 0

        if use_transformed_features:
            if predict_proba:
                try:
                    aucs.append(roc_auc_score(y_test, ada.fit(X_train_ft, y_train_contaminated).predict_proba(X_test_ft)[:, 1]))
                except AttributeError:
                    aucs.append(roc_auc_score(y_test, ada.fit(X_train_ft, y_train_contaminated).predict(X_test_ft)))
            else:
                aucs.append(roc_auc_score(y_test, ada.fit(X_train_ft, y_train_contaminated).predict(X_test_ft)))
        else:
            if predict_proba:
                try:
                    aucs.append(roc_auc_score(y_test, ada.fit(X_train, y_train_contaminated).predict_proba(X_test)[:, 1]))
                except AttributeError:
                    aucs.append(roc_auc_score(y_test, ada.fit(X_train, y_train_contaminated).predict(X_test)))
            else:
                aucs.append(roc_auc_score(y_test, ada.fit(X_train, y_train_contaminated).predict(X_test)))
    
    avg_auc = np.mean(aucs)
    auc_std = np.std(aucs)
    max_auc = np.max(aucs)
    min_auc = np.min(aucs)
    return (avg_auc, auc_std, min_auc, max_auc)

def benchmark_letter(ada, random_state=None, use_transformed_features=False, predict_proba=False):
    '''
    Benchmark the "letter" dataset as in the paper
    '''
    return _benchmark(ada, dataset='letter', random_state=random_state, use_transformed_features=use_transformed_features, predict_proba=predict_proba)
        
def benchmark_speech(ada, random_state=None, use_transformed_features=False, predict_proba=False):
    '''
    Benchmark the "speech" dataset as in the paper
    '''
    return _benchmark(ada, dataset='speech', random_state=random_state, use_transformed_features=use_transformed_features, predict_proba=predict_proba)
