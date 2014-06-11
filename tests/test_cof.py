import os

import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..lof import COF

def test_cof():
    k=7
    data = np.hstack(([np.arange(1, 11)[:, None], np.random.rand(10,1)*0.1]))
    data = np.vstack((data, [5, 1.5]))
    data = np.vstack((data, 
                      np.hstack((np.append(np.arange(1,10,1.5), 10)[:,None],
                                 np.random.rand(7,1)*0.2 - 10))))
    
    assert_array_almost_equal(COF(k=k).fit().predict(data),
                              np.array([0.9874, 0.9874, 0.9875, 0.9872, 0.9871, 0.9869, 0.9868, 0.9868, 0.9868, 0.9868, 1.0918, 1.0692, 1.0642, 1.0642, 1.0660, 1.0653, 1.0673, 1.0690]),
                              decimal=2)