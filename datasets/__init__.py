'''
Helper functions to load the synthetic datasets from
"Kernel PCA for novelty detection" Hoffmann, Heiko 
http://www.heikohoffmann.de/kpca.html
'''

from os.path import dirname
from os.path import join

import numpy as np

CURRENT_DIR = dirname(__file__)

def load_square():
    return np.loadtxt(join(CURRENT_DIR, 'square.dat'))
    
def load_square_noise():
    return np.loadtxt(join(CURRENT_DIR, 'square-noise.dat'))
    
def load_spiral():
    return np.loadtxt(join(CURRENT_DIR, 'spiral.dat'))
    
def load_sine_noise():
    return np.loadtxt(join(CURRENT_DIR, 'sine-noise.dat'))

def load_ring_line_square():
    return np.loadtxt(join(CURRENT_DIR, 'ring-line-square.dat'))