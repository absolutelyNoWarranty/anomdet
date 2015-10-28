import os
import scipy.io as sio

from ..base import OutlierDataset

PATH_TO_DATA = os.path.join(os.path.dirname(__file__), 'one_class_datasets')

def _load_oc_dataset(id_no):
    filename = 'oc' + '_' + str(id_no)
    ml_obj = sio.loadmat(os.path.join(PATH_TO_DATA, filename))
    
    # Kind of messy
    # Extract 'data' (the data) and 'nlab' (the outlier labels)
    X = ml_obj['x']['data'][0][0].astype('float')
    y = ml_obj['x']['nlab'][0][0] == 1
    y = y.flatten()
    return (X, y)
    
def load_breast_benign():
    '''
    Description:
        X : data
            458 target objects (originally the 'benign' class)
            241 outlier objects
            9 features
        
        y : labels, 'True' if outlier   
    '''
    X, y = _load_oc_dataset(505)
    return OutlierDataset(X, y, name="breast", pos_class_name="malignant",
                          neg_class_name="benign")
    
def load_heart_healthy():
    '''
    Description:
        X : data
            165 target objects (originally the 'absent' class)
            139 outlier objects
            13 features
        
        y : labels, True if outlier   
    '''
    X, y = _load_oc_dataset(507)
    return OutlierDataset(X, y, name="heart", pos_class_name="abnormal",
                          neg_class_name="normal")
    
def load_diabetes_absent():
    '''
    Description:
        X : data
            268 target objects (originally the 'absent' class)
            500 outlier objects
            8 features
        
        y : labels, True if outlier   
        
        
    Original UCI Dataset:
        Pima Indians Diabetes Database
    '''
    X, y = _load_oc_dataset(518)
    return OutlierDataset(X, y, name="diabetes", pos_class_name="diabetes",
                          neg_class_name="absent")
    
def load_arrhythmia_normal():
    '''
    Description:
        X : data
            237 target objects (originally the '1' class)
            183 outlier objects
            278 features
        
        y : labels, True if outlier   
        
        
    Original UCI Dataset:
        Arrhythmia
    '''
    X, y = _load_oc_dataset(514)
    return OutlierDataset(X, y, name="arrhythmia", pos_class_name="arrhythmia",
                          neg_class_name="normal")
                          
def load_hepatitis_normal():
    '''
    Description:
        X : data
            123 target objects (originally the 'live' class)
            32 outlier objects
            19 features
        
        y : labels, True if outlier   
        
        
    Original UCI Dataset:
        hepatitis
    '''
    X, y = _load_oc_dataset(516)
    return OutlierDataset(X, y, name="hepatitis", pos_class_name="die",
                          neg_class_name="live")
    
def load_colon_normal():
    '''
    Description:
        X : data
            22 target objects (originally the 'normal tissue' class)
            40 outlier objects
            1908 features
        y : labels, True if outlier
    '''
    X, y = _load_oc_dataset(570)
    return OutlierDataset(X, y, name="colon", pos_class_name="tumor",
                          neg_class_name="normal")