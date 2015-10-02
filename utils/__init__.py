import numpy as np
from CondensedDistanceMatrix import pdist_wrapper, CondensedDistanceMatrix
from normalize_scores import normalize_scores, replace_invalid_scores
from regularize_scores import regularize_scores

def unique_rows(ar, *args, **kwargs):
    '''
    Reference: http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array
    '''
    try:
        assert len(ar.shape) == 2
    except AssertionError:
        raise Warning("Expected a 2d array. Returning input unchanged.")
        return ar
    ar = np.ascontiguousarray(ar)
    results = np.unique(ar.view([('', ar.dtype)]*ar.shape[1]), *args, **kwargs)
    
    if isinstance(results, tuple):
        unique_ar = results[0]
        unique_ar = unique_ar.view(ar.dtype).reshape((unique_ar.shape[0], ar.shape[1]))
        
        return (unique_ar,) + results[1:]
    else:
        unique_ar = results
        return unique_ar.view(ar.dtype).reshape((unique_ar.shape[0], ar.shape[1]))
    