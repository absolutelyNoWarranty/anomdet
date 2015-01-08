import os

import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from .. import ABOD

def test_abod():
    '''
    Test ABOD with a simple toy dataset.
    '''
    A = np.array([[1, 0], [0, 1], [0, 0], [-1, 0], [0, -1], [5, 5]])
    
    assert_array_almost_equal(ABOD().fit().predict(A),
                              np.array([0.2431, 0.2431, 1.0000, 0.1894, 0.1894, 0]),
                              decimal=3)
