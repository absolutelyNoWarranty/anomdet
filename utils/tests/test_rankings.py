import os

import numpy as np

from sklearn.utils.testing import assert_array_equal


from ..rankings import scores_to_ranks, rank_distances

def test_scores_to_ranks():
    '''
    Test scores_to_ranks
    '''
    
    A = np.array([
        [0.9, 0.1, 0.2],
        [0.7, 0.3, 0.5],
        [0.2, 0.85, 0.9],
        [0.4, 0.15, 0.01]
    ])
    
    expected1 = np.array([
        [0, 3, 2],
        [1, 1, 1],
        [3, 0, 0],
        [2, 2, 3]
    ])
    
    assert_array_equal(scores_to_ranks(A), expected1)
    
    expected2 = np.array([
        [3, 0, 1],
        [2, 2, 2],
        [0, 3, 3],
        [1, 1, 0]
    ])
    
    assert_array_equal(scores_to_ranks(A, invert=True), expected2)

def test_rank_distances():
    '''
    Test rank_distances
    '''
    ranks = np.array([
        [0, 2, 0, 3],
        [3, 0, 1, 2],
        [1, 1, 2, 1],
        [2, 3, 3, 0],
        [4, 4, 4, 4]
    ])
    
    expected = np.array([
        [0, 6, 4, 6],
        [6, 0, 4, 6],
        [4, 4, 0, 8],
        [6, 6, 8, 0]
    ])
    
    assert_array_equal(rank_distances(ranks), expected)