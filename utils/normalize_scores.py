import numpy as np

def normalize_scores(X, method='uniform'):
    '''
    Normalize outlier scores
    Inputs:
        X : matrix of outlier scores, each column being a set of scores from one same method
        method : (default: 'uniform')
            'uniform' - a linear transformation to the range [0, 1]
            'gaussian' - gaussian distribution
            'gamma' - gamma distribution 
            'f' - F-distribution
    Output: the normalized scores
    
    See: Kriegel et al. "Interpreting and Unifying Outlier Scores"
    '''
    
    if method == 'uniform':
        # ignore nans
        mins = np.nanmin(X, axis=0)
        maxs = np.nanmax(X, axis=0)
        ranges = maxs-mins
        zero = np.where(ranges == 0.)[0]
        if len(zero):
            raise Exception("Range 0! Columns: %s" % str(list(zero)))
        norm_scores = (X - mins) / (ranges)
    
    return norm_scores
