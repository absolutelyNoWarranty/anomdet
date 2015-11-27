import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..base import Dataset, OutlierDataset

X = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
    [4, 4],
    [10, -10]
])

y1 = np.array(["a", "b", "c", "d", "e"])
y2 = np.array(["a", "a", "b", "b", "b"])
y3 = np.array([0, 1, 2, 1, 2])
y4 = np.array([True, False, True, False, True])

def test_dataset():
    '''
    Test Dataset
    '''
    
    D = Dataset(X, y=y1, name="Test Dataset")
    assert D.num_classes == 5
    
    D = Dataset(X, y=y2)
    assert D.num_classes == 2 
    
    
    D = Dataset(X, y=y3, classes_ = {0:'alpha', 1:'beta', 2:'gamma'})
    assert D._to_class_repr('alpha') == 0
    assert D._to_class_repr('beta') == 1
    assert D._to_class_repr('gamma') == 2
    assert_equal(D._to_class_repr(['gamma', 'beta', 'alpha']), [2, 1, 0])
    assert_equal(D._to_class_repr([0, 1, 2]), [0, 1, 2])
    
    assert_raises(ValueError, D._to_class_repr, 'dog')
    assert_raises(ValueError, D._to_class_repr, -1)
    
    assert_raises(TypeError, Dataset.__init__, X, y1, ['alp', 'bet', 'gam'])
    
def test_dataset_create_outlier_dataset():
    '''
    Test the create_outlier_dataset method of Dataset
    '''
    D = Dataset(X, y=y2)
    out_dat = D.create_outlier_dataset(n_to_select=[0.5, 'all'], normal_classes=['b'])
    assert out_dat.X.shape[0] == 4
    assert sum(out_dat.y) == 1
    
    out_dat = D.create_outlier_dataset(n_to_select=['all', 1./3], outlier_classes=['b'])
    assert out_dat.X.shape[0] == 3
    assert sum(out_dat.y) == 1
    
    out_dat1 = D.create_outlier_dataset(n_to_select=[1,1],
                                        outlier_classes=['b'],
                                        random_state=123)
    out_dat2 = D.create_outlier_dataset(n_to_select=[1,1],
                                        outlier_classes=['b'],
                                        random_state=123)
    assert_array_equal(out_dat1.X, out_dat2.X)
    assert_array_equal(out_dat1.y, out_dat2.y)
    
    out_dat1 = D.create_outlier_dataset(random_state=123)
    out_dat2 = D.create_outlier_dataset(random_state=123)
    assert_array_equal(out_dat1.X, out_dat2.X)
    assert_array_equal(out_dat1.y, out_dat2.y)
    
    
def test_dataset_show_instance():
    '''
    Test the show_instance method of Dataset
    '''
    D = Dataset(X, y=y1)
    expected_output = ("|   X0 |   X1 | Label   |\n"
                      "|-----:|-----:|:--------|\n"
                      "|    1 |    1 | a       |")

    assert D.instance_repr(0) == expected_output