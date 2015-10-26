from os.path import dirname
from os.path import join

import numpy as np

from .base import OutlierDataset

CURRENT_DIR = dirname(__file__)


'''
Helper functions to load the synthetic datasets from
"Kernel PCA for novelty detection" Hoffmann, Heiko 
http://www.heikohoffmann.de/kpca.html
'''

def simple_load_from_current_dir(filename, num_outliers):
    '''
    Loads .dat file from this directory.
    Assumes the last `num_outliers` are the outliers.
    '''
    X = np.loadtxt(join(CURRENT_DIR, filename))
    y = np.zeros(X.shape[0], dtype=bool)
    y[-num_outliers:] = True
    name = filename.split('.')[0]
    
    return OutlierDataset(X, y, name=name)

def load_square():
    return simple_load_from_current_dir('square.dat', 0)

    
def load_square_noise():
    return simple_load_from_current_dir('square-noise.dat', 50)
    
def load_spiral():
    return simple_load_from_current_dir('spiral.dat', 0)
    
def load_sine_noise():
    return simple_load_from_current_dir('spiral.dat', 200)
    
def load_ring_line_square():
    return simple_load_from_current_dir('ring-line-square.dat', 0)