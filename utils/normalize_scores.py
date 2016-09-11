import numpy as np

def replace_invalid_scores(X, nan_to='avg', inf_to='avg', neg_inf=True):
    '''
    Replace invalid values in a vector of outlier scores
    
    Parameters
    ----------
    X : array-like, shape = [n_samples,] or [n_samples, n_cols]
        Outlier scores, or matrix of outlier scores with each column being one
        array-like of outlier scores.

    nan_to : float or {'avg'} or list, length=n_cols, optional (default='avg')
        What to replace nan's with. If "avg", replace with average of
        (the respective column) of X.
    
    inf_to : float or {'avg'} or list, length=n_cols, optional (default='avg')
        What to replace nan's with. If "avg", replace with average of
        (the respective column) of X.
    
    neg_inf : bool, optional (default=True)
        Whether to treat negative infinity as negative. If true, replace -Inf
        with -inf_to.
    
    Returns
    -------
    A : X with nan's and inf's replaced with specified replacements.
        
    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(25.).reshape(5,5)
    >>> from anomdet.utils.normalize_scores import replace_invalid_scores
    >>> X[0, 0] = np.inf
    >>> X[1, 1] = np.nan
    >>> X[2:, 2] = -np.inf
    >>> replace_invalid_scores(X, nan_to='avg', inf_to=1000)
    array([[ 1000.  ,     1.  ,     2.  ,     3.  ,     4.  ],
           [    5.  ,    12.25,     7.  ,     8.  ,     9.  ],
           [   10.  ,    11.  , -1000.  ,    13.  ,    14.  ],
           [   15.  ,    16.  , -1000.  ,    18.  ,    19.  ],
           [   20.  ,    21.  , -1000.  ,    23.  ,    24.  ]])
    
    See also
    --------
    normalize_scores
    '''
    
    def _replace_invalid_scores(x, nan_to, inf_to):
        x = x.copy()
        
        if nan_to == 'avg':
            nan_to = np.ma.masked_invalid(x).mean()
        if inf_to == 'avg':
            inf_to = np.ma.masked_invalid(x).mean()
            
        x[np.isposinf(x)] = inf_to
        x[np.isneginf(x)] = -inf_to if neg_inf else inf_to
        x[np.isnan(x)] = nan_to
        return x
    
    if len(X.shape) == 1:
        return _replace_invalid_scores(X, nan_to=nan_to, inf_to=inf_to)
    
    if type(nan_to) != list:
        nan_to = [nan_to]*X.shape[1]
        
    if type(inf_to) != list:
        inf_to = [inf_to]*X.shape[1]
    
    scores = np.empty(X.shape)
    for j in xrange(X.shape[1]):
        scores[:, j] = (_replace_invalid_scores(X[:, j], nan_to[j], inf_to[j]))
    
    return scores
    
    
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
    
    Reference: Kriegel et al. "Interpreting and Unifying Outlier Scores"
    
    See also
    --------
    replace_invalid_scores
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
