import numpy as np
from sklearn.utils import check_random_state

def get_subsample_indices(classes, y, select, replace=False, random_state=None):
    '''
    Used to selectively sample from the classes of X.
    Takes the classes, list of labels and the number per class to select.
    Returns indices which can be used to subsample X.
    
    Parameters
    ----------
    `classes` : list or array-like
        The classes of `y` which are to be sampled from.
        
    `y` : array-like, shape (n_samples,)
        The class labels of some dataset X.
        
    `select` : array-like, shape(`classes`, ) 
        The number to select from each class.
        Elements can be:
            int, the exact number
            float, percentage
            "all", all
        
    `replace` : boolean, optional (default=False)
        Whether to sample with replacement or without.
        
    `random_state` : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    indices : ndarray
    Index array to use for subsampling.    
    '''
    
    random_state = check_random_state(random_state)
    
    result = []
    for num_to_select, class_ in zip(select, classes):
        indices = np.where(y==class_)[0]
        
        if num_to_select == "all":
            result.append(indices)
            continue
            
        if 0 <= num_to_select < 1.0:
            num_to_select = int(np.round(num_to_select * len(indices)))
        else:
            pass        
        result.append(random_state.choice(indices, num_to_select, replace=replace))
    result = np.concatenate(result)
    
    # Mix up the order of the indices to prevent user from accidently relying 
    # on an implicit ordering of the sample with regard to the classes
    n = len(result)
    result = result[random_state.choice(n, n, replace=False)]
    
    return result

