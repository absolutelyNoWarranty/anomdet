from sklearn.utils.testing import assert_equal
from ...datasets import load_thoraric_surgery

def test_thoraric_surgery():
    '''
    Test that load_thoraric_surgery works.
    '''
    dat = load_thoraric_surgery()
    assert_equal(sum(dat.y), 70)
    assert_equal(dat.name, 'thoraric_surgery')