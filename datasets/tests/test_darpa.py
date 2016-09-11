from ...datasets.darpa import load_darpa_data
from sklearn.utils.testing import assert_array_equal

def test_darpa_no_args():
    '''
    Test that load_darpa_data works with no args.
    '''
    dat = load_darpa_data()
    dat2 = load_darpa_data('all')
    assert_array_equal(dat.X, dat2.X)

def test_darpa_with_arg():
    '''
    Test that load_darpa_data works with one arg.
    '''
    dat = load_darpa_data('ipsweep')
    assert dat.X.shape == (76907L, 38L)