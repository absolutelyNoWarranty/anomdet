import os

import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..pca import PrincipalComponentReconstructionError

def test_principle_component_reconstruction_error():
    '''
    Test LOF with a simple toy dataset.
    '''
    A = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [0, -1.1], [0, -1.2], [0, -1.3]])
    noise = np.array([[0.0709, 0.0960], [0.0755, 0.0340], [0.0276, 0.0585], [0.0680, 0.0224], 
                      [0.0655, 0.0751], [0.0163, 0.0255], [0.0119, 0.0506], [0.0498, 0.0699]])
    
    pca_re = PrincipalComponentReconstructionError(n_components=2)
    
    pca_re.fit(A).predict(A+noise)