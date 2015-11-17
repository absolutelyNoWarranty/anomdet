from sklearn.utils.testing import assert_equal, assert_raises
from ..check_n_neighbors import check_n_neighbors

def test_check_n_neighbors():
    assert_equal(check_n_neighbors(3, 10), 3)
    assert_equal(check_n_neighbors(7, 10), 7)
    
    assert_equal(check_n_neighbors(3, 10), 3)

    assert_equal(check_n_neighbors(0.3, 10), 3)
    assert_equal(check_n_neighbors(0.5, 10), 5)
    
    assert_equal(check_n_neighbors(0.1, 25), 2) # 2.5 rounds down to nearest even number 2.0
    assert_equal(check_n_neighbors(0.1, 26), 3) # 2.5 rounds down to nearest even number 2.0
    
    assert_equal(check_n_neighbors(0.66, 31), 20)
    
    
    assert_raises(ValueError, check_n_neighbors, [1, 2], 10)
    
    assert_raises(ValueError, check_n_neighbors, 11, 10)
    assert_raises(ValueError, check_n_neighbors, -1, 10)
    assert_raises(ValueError, check_n_neighbors, 0, 10)
    assert_raises(ValueError, check_n_neighbors, 1.2, 10)
    assert_raises(ValueError, check_n_neighbors, 3.5, 10)
    
    assert_raises(ValueError, check_n_neighbors, 0.001, 25)