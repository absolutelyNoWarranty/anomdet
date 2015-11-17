import numpy as np
import warnings
from CondensedDistanceMatrix import pdist_wrapper, CondensedDistanceMatrix
from normalize_scores import normalize_scores, replace_invalid_scores
from regularize_scores import regularize_scores
from sklearn.utils import check_random_state

from simple_timer import SimpleTimer, my_timer
from .rankings import scores_to_ranks, rank_distances
from .check_n_neighbors import check_n_neighbors

DEFAULT_SEED = 888

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

def maybe_default_random_state(random_state):
    if random_state == None:
        warning_str = "No random_state given. Using seed {0}."
        warning_str = warning_str.format(DEFAULT_SEED)
        warnings.warn(warning_str)
        random_state = DEFAULT_SEED
    
    return check_random_state(random_state)