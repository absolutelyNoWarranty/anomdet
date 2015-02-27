import numpy as np

def regularize_scores(x, baseline, inversion_type=None):
    '''
    Regularize anomaly scores
    Inputs:
        scores : vector of anomaly scores
        baseline : The expected inlier values
        inversion_type : The type of inversion to use can be
            'linear' - linear inversion
            'log' - logarithmic  inversion
            None - don't invert
    '''
    
    if inversion_type == 'linear':
        x = np.max(x) - x
    elif inversion_type == 'log':
        x = -np.log(x/np.max(x))
        
    reg_scores = x - baseline
    reg_scores[reg_scores < 0] = 0
    
    return reg_scores