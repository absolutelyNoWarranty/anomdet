from sklearn.utils.testing import assert_equal

from ...datasets import load_forest_type

def test_forest_type():
    '''
    Test that load_forest_type works.
    '''
    D = load_forest_type()
    assert_equal((sum(D.y == 's') + sum(D.y == 'h') +
                  sum(D.y == 'd') + sum(D.y == 'o')),
                 len(D.y))