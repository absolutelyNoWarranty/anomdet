import os
import numpy as np

from .base import OutlierDataset

def load_aloi():
    X = np.loadtxt(os.path.join(os.path.dirname(__file__), 'aloi-27d-50000-max5-tot1508.csv.gz'),
                   usecols=range(27))
    y = 'Outlier' == np.loadtxt(os.path.join(os.path.dirname(__file__), 'aloi-27d-50000-max5-tot1508.csv.gz'),
                     usecols=[29],
                     dtype='str')
    
    return OutlierDataset(X, y=y, name='ALOI', allow_duplicates=False)