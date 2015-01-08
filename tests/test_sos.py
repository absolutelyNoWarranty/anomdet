import os

import numpy as np

from sklearn.datasets import load_iris

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..sos import StochasticOutlierSelection


HERE = os.path.dirname(__file__)

def test_sos():
    '''
    Test stochastic outlier selection
    '''
    iris = load_iris()
    sos_probs = StochasticOutlierSelection().fit().predict(iris.data)
    sos_iris_probs = np.loadtxt(os.path.join(HERE,'iris_sos_results'))
    print np.sqrt(np.mean((sos_probs - sos_iris_probs)**2))
    random_probs = sos_iris_probs[np.random.choice(150,150,replace=False)]
    print np.sqrt(np.mean((sos_probs - random_probs)**2))
    print np.sqrt(np.mean((sos_probs - np.random.rand(150))**2))
    print np.argsort(sos_iris_probs)
    print np.argsort(sos_probs)
    assert_array_almost_equal(sos_probs, sos_iris_probs) 