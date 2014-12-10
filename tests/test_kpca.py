import os

import numpy as np

from sklearn.utils.testing import assert_almost_equal

from ..kpca import KPCA
from ..datasets import load_square, load_square_noise, load_spiral, load_sine_noise, load_ring_line_square


HERE = os.path.dirname(__file__)

def test_kpca_square():
    '''
    Test KPCA with square.dat
    '''
    X = load_square()
    recerr = KPCA(sigma=0.1, n_eigval=2).fit(X).predict(X)
    matlab_recerr = np.loadtxt(os.path.join(HERE, 'square.out'))
    assert_almost_equal(recerr, matlab_recerr, decimal=4)
    
def test_kpca_square_noise():
    '''
    Test KPCA with square-noise.dat
    '''
    X = load_square_noise()
    recerr = KPCA(sigma=0.1, n_eigval=2).fit(X).predict(X)
    matlab_recerr = np.loadtxt(os.path.join(HERE, 'square-noise.out'))
    assert_almost_equal(recerr, matlab_recerr, decimal=4)
    
def test_kpca_spiral():
    '''
    Test KPCA with spiral.dat
    '''
    X = load_spiral()
    recerr = KPCA(sigma=0.1, n_eigval=2).fit(X).predict(X)
    matlab_recerr = np.loadtxt(os.path.join(HERE, 'spiral.out'))
    assert_almost_equal(recerr, matlab_recerr, decimal=4)
    
def test_kpca_sine_noise():
    '''
    Test KPCA with sine-noise.dat
    '''
    X = load_sine_noise()
    recerr = KPCA(sigma=0.1, n_eigval=2).fit(X).predict(X)
    matlab_recerr = np.loadtxt(os.path.join(HERE, 'sine-noise.out'))
    assert_almost_equal(recerr, matlab_recerr, decimal=4)
    
def test_kpca_ring_line_square():
    '''
    Test KPCA with ring-line-square.dat
    '''
    X = load_ring_line_square()
    recerr = KPCA(sigma=0.1, n_eigval=2).fit(X).predict(X)
    matlab_recerr = np.loadtxt(os.path.join(HERE, 'ring-line-square.out'))
    assert_almost_equal(recerr, matlab_recerr, decimal=4)