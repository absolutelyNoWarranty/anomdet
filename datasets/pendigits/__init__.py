import os
import numpy as np
import matplotlib.pyplot as plt

from ..dataset import Dataset

# absolute path to directory this source file is in
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_pendigits():
    '''
    Original dataset: UCI Pendigits: "pendigits.tra"
    '''
    X = np.loadtxt(os.path.join(THIS_DIR, "pendigits.tra"), delimiter=",")
    y = X[:, -1]
    X = X[:, :-1].astype(int)
    
    return (X, y) 

def show_pendigit(vals, ax=None):
    if ax is None:
        ax = plt.gca()
    
    vals = np.array(vals).reshape(8, 2)
    
    #vals[:, 1] = 100 - vals[:, 1]
    
    ax.plot(vals[:, 0], vals[:, 1], marker="o")
    
    x_from = vals[0, 0]
    y_from = vals[0, 1]
    for i in range(1,8):
        x_to = vals[i, 0]
        y_to = vals[i, 1]
        
        vec_x = x_to - x_from
        vec_y = y_to - y_from
        
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from), arrowprops=dict(frac=0.1,headwidth=10., width=2.))
        x_from = x_to
        y_from = y_to
    
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))
    
    return ax
    
def load_pendigits():
    features = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8"]
    
    (X, y) = _load_pendigits()
    
    return Dataset(X, y, name="pendigits", features=features,
                   show_instance_f=show_pendigit)