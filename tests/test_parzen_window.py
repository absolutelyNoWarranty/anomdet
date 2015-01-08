import os

import numpy as np

from sklearn.utils.testing import assert_almost_equal

from ..parzen_window import ParzenWindow
from ..datasets.digits import load_digits_data
#from ..datasets import load_square, load_square_noise, load_spiral, load_sine_noise, load_ring_line_square


HERE = os.path.dirname(__file__)

def test_pw_runs():
    '''
    Test to see parzen windows runs
    '''
    X,y = load_digits_data([990,10,0,0,0,0,0,0,0,0], random_state=123)
    clf = ParzenWindow().fit(X).predict(X)
    
