import numpy as np

def normalize_scores(X, method='uniform', best_effort=True):
    '''
    Normalize outlier scores
    Inputs:
        X : matrix of outlier scores, each column being a set of scores from one same method
        method : (default: 'uniform')
            'uniform' - a linear transformation to the range [0, 1]
            'gaussian' - gaussian distribution
            'gamma' - gamma distribution 
            'f' - F-distribution
        best_effort : Don't raise exception
    Output: the normalized scores
    
    See: Kriegel et al. "Interpreting and Unifying Outlier Scores"
    '''
    
    if method == 'uniform':
        # ignore nans
        mins = np.array(np.min(np.ma.masked_invalid(X), axis=0))
        maxs = np.array(np.max(np.ma.masked_invalid(X), axis=0))
        ranges = maxs-mins
        zero = np.where(ranges == 0.)[0]
        if len(zero):
            if best_effort:
                ranges[zero] = 1.0
                # will turn everything to 0 since it's minus mins which is the same for everything
                #raise Warning("Range is 0! Columns: %s" % str(list(zero)))
            else:
                raise Exception("Range is 0! Columns: %s" % str(list(zero)))
            
        norm_scores = (X - mins) / (ranges)
    
    return norm_scores
