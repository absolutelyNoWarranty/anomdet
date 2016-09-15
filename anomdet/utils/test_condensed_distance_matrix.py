import os

import numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises


from CondensedDistanceMatrix import pdist_wrapper


def test_pdist_wrapper_and_CondensedDistanceMatrix():
    A = np.random.rand(8, 2)
    
    distance_matrix = squareform(pdist(A))
    distance_matrix_wrapper = pdist_wrapper(A)
    #print distance_matrix
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            print i,j
            assert_equal(distance_matrix[i, j], distance_matrix_wrapper[i, j])
    
    assert_raises(IndexError, distance_matrix_wrapper.__getitem__, (8, 0))
    assert_raises(IndexError, distance_matrix_wrapper.__getitem__, (0, 8))
    assert_raises(IndexError, distance_matrix_wrapper.__getitem__, (-1, 0))
    assert_raises(IndexError, distance_matrix_wrapper.__getitem__, (0, -1))

def test_pdist_wrapper_and_CondensedDistanceMatrix2():
    A = np.random.rand(100, 50)
    
    distance_matrix = squareform(pdist(A))
    distance_matrix_wrapper = pdist_wrapper(A)

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            print i,j
            assert_equal(distance_matrix[i, j], distance_matrix_wrapper[i, j])