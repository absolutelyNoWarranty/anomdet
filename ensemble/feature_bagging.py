import numpy as np
from sklearn.utils import check_random_state
from combine_scores import combine_scores

def feature_bagging(X, outlier_scorer, n_rounds, k=None, return_all_scores=False, random_state=None):
    '''
    Inputs:
        X - data, m by n feature matrix, each row is an instance
        outlier_scorer - an outlier scoring function
        k - number of features per feature bag
            if k is false or 0, then will randomly chose a number between
            n/2 and n every round
        n_rounds - number of rounds (number of feature bags generated)
        return_all_scores - bool, if True return scores generated each round,
            otherwise return average (default : False)
        random_state : int or instance of RandomState
            default : None
    '''
    random_state = check_random_state(None)
    
    m, n = X.shape
    
    scores = np.empty(shape=(m, n_rounds), dtype=float)
    for i in range(n_rounds):
        if k > 0:
            featind = random_state.choice(n, k, replace=False)
        else:
            featind = random_state.choice(n, random_state.randint(np.floor(n/2), n))
        scores[:, i] = outlier_scorer(X[:, featind])

    if return_all_scores:
        return scores
        
    return combine_scores(scores, method='avg')
    
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from anomdet import LOF
    X = load_iris().data
    f = LOF(k=4).predict
    
    print feature_bagging(X, f, n_rounds=3)
    