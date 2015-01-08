import os

import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..lof import LOF

def test_lof():
    '''
    Test LOF with a simple toy dataset.
    '''
    A = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, -1.1], [0, -1.2], [0, -1.3]])
    noise = np.array([[0.0709, 0.0960], [0.0755, 0.0340], [0.0276, 0.0585], [0.0680, 0.0224], 
                      [0.0655, 0.0751], [0.0163, 0.0255], [0.0119, 0.0506], [0.0498, 0.0699]])
    
    here = os.path.dirname(__file__)
    for k in range(1,8):
        answer = np.genfromtxt(os.path.join(here, "lof_matlab_k_%d.csv" % k))
        assert_array_almost_equal(LOF(k=k).fit().predict(A+noise),
                                  answer,
                                  decimal=3)

def test_lof_api():
    '''
    Test LOF with a simple toy dataset. Test that both usages work.
    '''
    A = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, -1.1], [0, -1.2], [0, -1.3]])
    noise = np.array([[0.0709, 0.0960], [0.0755, 0.0340], [0.0276, 0.0585], [0.0680, 0.0224], 
                      [0.0655, 0.0751], [0.0163, 0.0255], [0.0119, 0.0506], [0.0498, 0.0699]])
    
    
    for k in range(1,8):
        lof1 = LOF(k=k)
        lof2 = LOF(k=k)
        assert_array_equal(lof1.fit().predict(A+noise), lof2.fit(A+noise).predict(A+noise))
        
# def test_lof_api2():
    # '''
    # Test LOF train/test
    # '''
    # A = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, -1.1], [0, -1.2], [0, -1.3]])
    # noise = np.array([[0.0709, 0.0960], [0.0755, 0.0340], [0.0276, 0.0585], [0.0680, 0.0224], 
                      # [0.0655, 0.0751], [0.0163, 0.0255], [0.0119, 0.0506], [0.0498, 0.0699]])
    # A = A + noise

    # for k in range(1,8):
        # lof1 = LOF(k=k).fit(A)
        # lof2 = LOF(k=k).fit(A[1:, :])
        # assert_array_equal(lof1.predict(A[0:2, :]), lof2.predict(A[0:2, :]))
            
def test_lof_error_checking():
    A = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, -1.1], [0, -1.2], [0, -1.3]])
    lof_8 = LOF(k=0)
    assert_raises(ValueError, lof_8.fit, A)

def test_lof2():
    '''
    Test LOF with a simple toy dataset which has neighbors with equal distances.
    TODO: MAKE THIS TEST WORK
    '''
    
    A = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, -1.1], [0, -1.2], [0, -1.3]])
    
    
    #assert_array_equal(LOF(k=1).fit().predict(A),
    #  np.array([3.2500, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]))
    
    #assert_array_equal(LOF(k=2).fit().predict(A),
    #  np.array([2.9547, 3.4938, 0.9912, 3.4938, 1.0000, 1.0000, 1.0000, 1.0000]))
    
    assert_array_equal(LOF(k=3).fit().predict(A),
      np.array([2.1746, 2.4809, 0.9912, 2.4809, 0.9167, 1.0952, 1.0952, 0.9167]))
      
    assert_array_equal(LOF(k=6).fit().predict(A),
      np.array([2.1746, 2.4809, 0.9912, 2.4809, 0.9167, 1.0952, 1.0952, 0.9167]))
    
    assert_array_equal(LOF(k=7).fit().predict(A),
      np.array([2.1746, 2.4809, 0.9912, 2.4809, 0.9167, 1.0952, 1.0952, 0.9167]))
    
    assert_raises(InputError, LOF, k=8)