import os
import numpy as np
from ..utils import unique_rows


def load_aloi(remove_duplicates=True):
    X = np.loadtxt(os.path.join(os.path.dirname(__file__), 'aloi-27d-50000-max5-tot1508.csv.gz'),
                   usecols=range(27))
    y = 'Outlier' == np.loadtxt(os.path.join(os.path.dirname(__file__), 'aloi-27d-50000-max5-tot1508.csv.gz'),
                     usecols=[29],
                     dtype='str')
    
    if remove_duplicates:
        X, ind = unique_rows(X, return_index=True)
        y = y[ind]
        
    return (X, y)