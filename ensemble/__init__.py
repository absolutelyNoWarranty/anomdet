import numpy as np
from scipy.special import erf

from combine_scores import combine_scores

def normalize_anomaly_scores(scores, scaling_type='uniform'):
    '''
    Regularize and normalize anomaly scores
    Inputs:
        `scores`: A vector of anomaly scores
        `normalization_type`: The type of scaling to use
            'uniform', or not given - the default, a linear transformation from
                              0 to 1
            'gaussian' - gaussian distribution UNIMPLEMENTED
            'gamma' - gamma distribution UNIMPLEMENTED 
            'cauchy' - Cauchy-distribution UNIMPLEMENTED
            'f' - F-distribution UNIMPLEMENTED
            not given - don't invert the scores
    Outputs: 
        `norm_scores`: The normalized scores
    Reference: Kriegel et al. "Interpreting and Unifying Outlier Scores"
    '''
    
    def regularize(scores, baseline=0, inversion_type=None):
        '''
        Regularize anomaly scores
        Inputs:
            scores: A vector of anomaly scores
            baseline: The expected inlier value
            inversion_type: The type of inversion to use, can be
                'linear'
                'log' - logarithmic inversion
                not given - don't invert the scores
        Outputs: 
                reg_scores: The regularized scores
        Reference: Kriegel et al. "Interpreting and Unifying Outlier Scores"
        '''
        if inversion_type == 'linear':
            scores = np.max(scores) - scores
        elif inversion_type == 'log':
            scores = -np.log(scores/max(scores))
            
        reg_scores = scores - baseline;
        reg_scores[reg_scores < 0] = 0;
    
    max_v = np.nanmax(scores)
    min_v = np.nanmin(scores)
    
    if scaling_type =='uniform':
        norm_scores = (scores - min_v) / (max_v - min_v)
    elif scaling_type == 'gaussian':
        mu = np.mean(scores)
        sigma = np.std(scores)
        norm_scores = erf((scores - mu) / (np.sqrt(2)*sigma))
        norm_scores[norm_scores < 0] = 0
    elif scaling_type == 'gamma':
        pass
    
    return norm_scores
    
def random_ensemble(scores, ensemble_size):
    '''
    Randomly construct an ensemble and take the mean. This should be used as a
    benchmark against other ensemble methods.
    '''
    pass
    
def feature_bagging(A, outlier_scorer, k, num_feature_sets):
    pass

def combine_outlier_scores():
    pass
def union_ensemble():
    pass

class AnomalyScoreCombiner(object):
    def __init__(self, scores):
        self.scores = scores
    
    def mean(self, normalize=True):
        norm_scores = []
        for j in range(self.scores.shape[1]):
            norm_scores.append(normalize_anomaly_scores(self.scores[:, j]))
        norm_scores = np.vstack(norm_scores).T
        return np.mean(norm_scores, axis=1)
    
    