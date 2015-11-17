import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..knn import KNN

def test_knn():
    '''
    Test KNN with a simple toy dataset.
    '''
    X1 = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0]])
    
    X2 = np.array([
        [0, 1],
        [1, -1],
        [2, 1],
        [3, -1],
        [4, 1],
        [5, -1]])
    
    assert_array_equal(KNN(k=1).fit(X1).predict(X1), np.ones(6, dtype=float))
    assert_array_equal(KNN(k=1).fit(X1).predict(X2), np.ones(6, dtype=float))
    
    assert_array_equal(KNN(k=2).fit(X1).predict(X1),
                       np.array([ 1.5,  1. ,  1. ,  1. ,  1. ,  1.5]))
                       
    assert_array_equal(KNN(k=2).fit(X1).predict(X2),
                       np.ones(6)*(np.sqrt(2)+1)/2.)

def test_knn_value_error():
    '''
    Test raises ValueError for improper values of k
    '''
    
    knn = KNN(k=20)
    rs = np.random.RandomState(0)
    assert_raises(ValueError, knn.fit, rs.rand(10, 5000))
    
    rs = np.random.RandomState(0)
    knn = KNN(k=0.001)
    assert_raises(ValueError, knn.fit, rs.rand(10, 5000)) 