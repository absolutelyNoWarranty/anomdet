import numpy as np
    
def check_n_neighbors(k, n):
    '''
    Utility for checking n_neighbors parameter.
    
    Parameters
    ----------
    k : int or float
        If 
        int - number of neighbors
        float - percentage of neighbors
        
    n : int
        total number of samples
        
    Returns
    -------
    n_neighbors : int
        The number of neighbors
        
    '''
    
    if isinstance(k, np.ndarray):
        k = np.asscalar(k)
    
    if isinstance(k, int):
        if k > n:
            raise ValueError("k is greater than the number of samples!")
        if k <= 0:
            raise ValueError("k is too small!")
        return k
    
    elif isinstance(k, float):
        if 0. < k < 1.:
            return check_n_neighbors(int(np.round(k * n)), n)
        else:
            raise ValueError("When float, k should be between 0 and 1.")
    else:
        raise ValueError("k should be an integer or float")
    