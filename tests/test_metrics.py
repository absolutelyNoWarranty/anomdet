import numpy as np

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_greater

from ..metrics import purity, entropy, precision, recall

# These clustering stats taken from Chapter 8 of Introduction to Data Mining (Tan, Steinbach, Kumar)
class_names = ['entertainment', 'financial', 'foreign', 'metro', 'national', 'sports']
clusters = [1,2,3,4,5,6]
clustering = np.array([[3, 5, 40, 506, 96, 27],
                       [4, 7, 280, 29, 39, 2],
                       [1, 1, 1, 7, 4, 671],
                       [10, 162, 3, 119, 73, 2],
                       [331, 22, 5, 70, 13, 23],
                       [5, 358, 12, 212, 48, 13]])

# Create a set of cluster labels and class labels based on the clustering          
class_labels=reduce(np.append, map(lambda (num, name): np.repeat(name, num), zip(np.sum(clustering,axis=0),class_names)))

cluster_labels = reduce(np.append, [reduce(np.append, map(lambda (cluster_id, num): np.repeat(cluster_id, num), enumerate(class_dist.tolist(), 1))) for class_dist in list(clustering.T)])


def test_purity():
    '''
    Test the purity metric
    '''
    purity_reference = [None, 0.7474, 0.7756, 0.9796, 0.4390, 0.7134, 0.5525, 0.7203]
    purest_classes_reference = [None, 'metro', 'foreign', 'sports', 'financial', 'entertainment', 'financial']
    
    for i in clusters:
        print i
        class_labels_in_this_cluster = class_labels[cluster_labels == i]
        purity_score, purest_class = purity(class_labels_in_this_cluster, return_class=True)
        assert_almost_equal(purity_score, purity_reference[i], decimal=4)
        assert_equal(purest_class, purest_classes_reference[i])

def test_entropy():
    '''
    Test the entropy metric
    '''

    entropy_reference = [None, 1.2270, 1.1472, 0.1813, 1.7487, 1.3976, 1.5523, 1.1450]
    
    for i in clusters:
        class_labels_in_this_cluster = class_labels[cluster_labels == i]
        assert_almost_equal(entropy(class_labels_in_this_cluster), entropy_reference[i], decimal=4)

def test_precision():
    assert_almost_equal(precision(cluster_labels, class_labels, "metro", 1), 0.75, decimal=2)

def test_recall():
    assert_equal(recall(cluster_labels, class_labels, "metro", 1), 506./943)