def normalize_scores(x, method='uniform'):
    '''
    Normalize outlier scores
    Inputs:
        scores : a vector of anomaly scores
        method : (default: 'uniform')
            'uniform' - a linear transformation to the range [0, 1]
            'gaussian' - gaussian distribution
            'gamma' - gamma distribution 
            'f' - F-distribution
    Output: the normalized scores
    
    See: Kriegel et al. "Interpreting and Unifying Outlier Scores"
    '''
    
    if method == 'uniform':
        norm_scores = (scores - np.min(x)) / (np.max(x) - np.min(x))
    
    return norm_scores
