import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def scores_to_ranks(matrix_of_scores, invert=False):
    '''
    Takes a matrix of scores and converts them to ranks.
    
    Parameters:
    -----------
    matrix_of_scores : array-like, shape (n_samples, n_score_lists)
    invert: : boolean, optional (default=False)
        If False, smaller rankings corresponding to higher scores. (0 is the highest scoring.)
        If True, larger rankings corresponding to higher scores. (0 is the lowest scoring.)
    
    Returns:
    --------
    matrix_of_ranks : array-like, shape (n_samples, n_score_lists)
    '''
    if not invert:
        return np.argsort(np.argsort(-matrix_of_scores, axis=0), axis=0)
    else:
        return np.argsort(np.argsort(matrix_of_scores, axis=0), axis=0)

def rank_distances(ranks, k=0.25):
    '''
    Parameters:
    ----------
    ranks : array-like, shape (n_samples, n_rank_lists)
        The i-th column is a independent set of ranks for the data consisting `n_samples` data.
        Ranks are assumed to be in the range [0, n_samples) with no repeated ranks.
        
    Returns:
    --------
    D : array [n_samples_a, n_samples_a] or [n_samples_a, n_samples_b]

    A distance matrix D such that D_{i, j} is the rank distance between the ith and jth column
    vectors of the given matrix X. 

    '''
    # Only calculate for the top 25% ranks
    k = int(np.round(k*ranks.shape[0]))
    
    def _metric(r1, r2):
        return np.abs(np.argsort(r1) - np.argsort(r2))[:k].sum()
        
            
    return pairwise_distances(X=ranks.T, metric=_metric)