# Thoracic Surgery Data Set
# Reference: https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data#

import os
import numpy as np
from scipy.io.arff import loadarff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ...datasets import Dataset

# absolute path to directory this source file is in
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
FILEPATH = os.path.join(THIS_DIR, "ThoraricSurgery.arff")

def _load_thoraric_surgery():
    '''
    Load "ThoraricSurgery.arff"
    '''
    # (alphabetically 'F' comes before 'T' so it's ok
    bool_l_enc = LabelEncoder().fit(['T', 'F']) 
    l_enc = LabelEncoder()
    
    arffdata = loadarff(FILEPATH)
    
    data = arffdata[0]
    info = arffdata[1]
    
    header = [k for k,v in eval(str(data.dtype))]
    #header.pop()
    
    X = np.empty((data.shape[0], len(header)))
    features = []
    categorical = []
    for j in range(len(header)):
        if data.dtype[j] == 'S4': # categorical
            X[:, j] = l_enc.fit_transform(data[header[j]])
            features.extend([header[j]+'-'+s for s in l_enc.classes_])
            categorical.append(j)
            
        elif data.dtype[j] == 'S1': # binary 
            X[:, j] = bool_l_enc.transform(data[header[j]])
            features.append(header[j])
            
        else:
            X[:, j] = data[header[j]]
            features.append(header[j])
    
    label_name = features.pop()
    
    # one-hot encoding
    oh_enc = OneHotEncoder(sparse=False, categorical_features=categorical)
    X = oh_enc.fit_transform(X)
    
    y = X[:, -1].astype(bool)
    X = X[:, :-1]
    
    return (X, y, features, label_name) 
    
def load_thoraric_surgery():
    (X, y, features, label_name) = _load_thoraric_surgery()
    classes_ = {False:'survived', True: 'died'}
    return Dataset(X, y, name="thoraric_surgery",
                   features=features, classes_=classes_)