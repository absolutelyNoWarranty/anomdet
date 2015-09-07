from anomdet.ensemble import combine_scores
import numpy as np

def proj_column_space(A, b):
    '''
    Returns the projection of b onto the column space of A
    '''
    
    return A.dot(np.linalg.inv(A.T.dot(A))).dot(A.T.dot(b))

def dist_to_col_space(A, b):
    try:
        p = proj_column_space(A, b)
        return np.linalg.norm(p - b)
    except LinAlgError:
        return np.sqrt(np.linalg.lstsq(A, b))[0]
        
def col_space_feature_selection(A, delta):
    '''
    Select columns of A based on maximizing distance to column
    space
    
    '''
    
    assert delta > 0
    
    # Start with 2 random columns
    n = A.shape[1]
    n_range = np.arange(n)
    available = np.ones(n).astype('bool') # available to be chosen
    init = np.random.choice(n, 2, replace=False)
    available[init] = False 
    num_remaining = n-2
   
    while num_remaining > 0:
        chosen = np.random.choice(num_remaining, 1)
        
        if dist_to_col_space(A[:, np.logical_not(available)], A[:, chosen]) > delta:
            num_remaining -= 1
            available[n_range[available][chosen]] = False
        else:
            break
    
    return A[:, np.logical_not(available)]

def col_space_feature_selection2(matrix_of_scores, **kwargs):
    '''
    Select columns of A based on maximizing distance to column
    space
    
    '''
    
    def weighted_pearson_correlation(u, v, w):
        '''
        Calculates the weighted (with weights `w`) pearson correlation between
        u and v
        
        TODO: make faster?
        '''
        # Normalize w so it sums to 1
        if np.sum(w) != 1.:
            w = w / np.sum(w)
        
        u = u.flatten()
        v = v.flatten()
        
        mean_u = w.dot(u)
        var_u = w.dot((u - mean_u)**2)
        
        mean_v = w.dot(v)
        var_v = w.dot((v - mean_v)**2)
        
        cov_uv = np.sum((u - mean_u)*(v - mean_v)*(w))
        rho = cov_uv / np.sqrt(var_u) / np.sqrt(var_v)
        
        return rho   
        
    k = kwargs.get('k', None)
    if k is None:
        raise Exception("Missing parameter k")
    ensemble_indices = kwargs.get('ensemble_indices', False)
    m, t = matrix_of_scores.shape
    
    # Create "target" vector (NOT used as the actual result)
    top_k_union = np.unique(np.argsort(-matrix_of_scores, axis=0)[:k])
    target_vec = np.zeros(m)
    target_vec[top_k_union] = 1
    K = len(top_k_union)
    
    # Weights for the weighted Pearson correlation
    w = np.empty(m)
    w[:] = K
    w[top_k_union] = m-K
    
    # Ensemble selection
    in_ensemble = np.repeat(False, t)
    detectors = set(range(t))
    
    # Initialize, find detector with highest weighted Pearson correlation 
    # to target vector
    corrs = [weighted_pearson_correlation(target_vec, matrix_of_scores[:, j], w) for j in range(t)]
    i = np.argmax(corrs)
    in_ensemble[i] = True
    detectors.discard(i)
    while detectors:
        detectors_ = list(detectors)
        
        curr_ensemble_output = combine_scores(matrix_of_scores[:, in_ensemble], method='avg')
       
        # Find detector with lowest column_space distance to the current ensemble
        col_space_dist= [dist_to_col_space(matrix_of_scores[:, in_ensemble], matrix_of_scores[:, j]) for j in detectors_]
        i = detectors_[np.argmin(col_space_dist)]
        
        # Decide whether to add this detector to the ensemble
        curr_ensemble_corr_with_target = weighted_pearson_correlation(target_vec, curr_ensemble_output, w)
        in_ensemble[i] = True
        new_ensemble_output = combine_scores(matrix_of_scores[:, in_ensemble], method='avg')
        new_ensemble_corr_with_target = weighted_pearson_correlation(target_vec, new_ensemble_output, w)
        if new_ensemble_corr_with_target <= curr_ensemble_corr_with_target:
            in_ensemble[i] = False
        
        detectors.discard(i)
    
    if ensemble_indices:
        return (combine_scores(matrix_of_scores[:, in_ensemble], method='avg'), in_ensemble)
    else:
        return combine_scores(matrix_of_scores[:, in_ensemble], method='avg')
    
    
    