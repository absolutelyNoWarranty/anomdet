import numpy as np

def entropy(labels):
    """Calculates the entropy for a single cluster, given the class labels of items in that cluster"""
    if len(labels) == 0:
        return 1.0
    label_idx = np.unique(labels, return_inverse=True)[1]
    pi = np.bincount(label_idx).astype(np.float)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return -np.sum((pi / pi_sum) * (np.log2(pi) - np.log2(pi_sum)))

def purity(labels, return_class=True):
    """purity
    Calculates the entropy for a single cluster, given the class labels of items in that cluster.
    If return_class is true, returns the purest class.
    """
    if len(labels) == 0:
        return 1.0
    (classes, label_idx) = np.unique(labels, return_inverse=True)
    pi = np.bincount(label_idx).astype(np.float)
    pi_sum = np.sum(pi)
    prob = pi/pi_sum
    
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    purest_ind = np.argmax(prob)
    return (prob[purest_ind], classes[purest_ind])
    
def precision(cluster_labels, class_labels, class_name, cluster_id):
    return np.mean(class_labels[cluster_labels==cluster_id] == class_name)

def recall(cluster_labels, class_labels, class_name, cluster_id):
    return np.mean(cluster_labels[class_labels==class_name] == cluster_id)
    
def f1(cluster_labels, class_labels, class_name, cluster_id):
    precision_ = precision(cluster_labels, class_labels, class_name, cluster_id)
    recall_ = recall(cluster_labels, class_labels, class_name, cluster_id)
    return 2 * precision_ * recall_ / (precision_ + recall_)