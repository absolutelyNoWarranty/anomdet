# Forest type mapping Data Set
# Reference: https://archive.ics.uci.edu/ml/datasets/Forest+type+mapping#

import os
import numpy as np

from ...datasets import Dataset

# absolute path to directory this source file is in
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(THIS_DIR, "training.csv")
TEST = os.path.join(THIS_DIR, "testing.csv")

def _load_forest_type():
    '''
    Load "training.csv" and "testing.csv" 
    '''

    header = open(TRAIN).readline().strip().split(",")
    features = header[1:]
    
    
    X_tra = np.loadtxt(TRAIN, skiprows=1, delimiter=",",
                       usecols=range(1, len(header)))
    y_tra = np.loadtxt(TRAIN, skiprows=1, delimiter=",", usecols=[0],
                       dtype='|S1')

    X_tst = np.loadtxt(TEST, skiprows=1, delimiter=",",
                       usecols=range(1, len(header)))
    y_tst = np.loadtxt(TEST, skiprows=1, delimiter=",", usecols=[0],
                       dtype='|S1')
    
    X = np.vstack([X_tra, X_tst])
    y = np.append(y_tra, y_tst)
    
    return (X, y, features) 
    
def load_forest_type():
    features = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8"]
    
    classes_ = dict(s="sugi", h="hinoki", d="deciduous", o="other")
    
    (X, y, features) = _load_forest_type()
    
    return Dataset(X, y, name="forest_type",
                   features=features, classes_=classes_)