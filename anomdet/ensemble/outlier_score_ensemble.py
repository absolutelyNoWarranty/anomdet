import numpy as np
from sklearn.externals.joblib import Parallel, delayed

def _fit_one(est, X):
    return est.fit(X)

def _predict_one(est, X):
    return est.predict(X)

def make_outlier_score_ensemble(X, estimators, n_jobs=1):
    '''
    Make a matrix of outlier scores.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
            Data
    
    estimators : list of (estimator) tuples
    
    n_jobs : int
    
    Returns
    -------
    outlier_scores : array, shape (n_sample, len(estimators))
    '''
    
    estimators = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_fit_one)(est, X)
        for est in estimators)
    
    outlier_scores = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_predict_one)(est, X)
        for est in estimators)
    
    outlier_scores = np.vstack(outlier_scores).T
    
    return outlier_scores