import numpy as np

def inverse_ranking(matrix_of_scores):
    '''Calculates the avg inverse ranking (harmonic mean right??)
    Input:
        matrix_of_scores - matrix of outlier scores where each column is a score-list of items
        
    Output:
        S - final outlier scores calculated after aggregating the ranklists
        finalrank - aggregated final ranklist
    '''
    
    n = matrix_of_scores[0]
    
    #matrix_of_rankings = np.argsort(np.argsort(matrix_of_scores, axis=0), axis=0) # larger is more anomalous, so 0 is least anomalous and n is most
    
    matrix_of_rankings = np.argsort(np.argsort(matrix_of_scores, axis=0)[::-1], axis=0) + 1 # smaller is more anomalous, so 1 is #1 ranked for anomalous

    return np.mean(1./matrix_of_rankings, axis=1)
