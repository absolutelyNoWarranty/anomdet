import pandas as pd
import numpy as np
import os
from sklearn.datasets.base import Bunch

def get_data_home():
    return './benchmark_datasets_original'

def get_data_path(name):
    return os.path.join(get_data_home(), name)
    
def load_magic_gamma_telescope():
    '''
    MAGIC Gamma Telescope
    19020 instances
    10 continuous attributes
    binary class
    "magic04.data"
    '''
    data = pd.read_csv(os.path.join(get_data_home(), 'magic04.data'), header=None)
    flat_data = data.ix[:,0:9]
    labels = data.ix[:, 10]
    
    return Bunch(data=flat_data.values,
                 target=labels.values,
                 name='magic',
                 dataset_type='classification')
                 
def load_miniboone_particle_identification():
    '''
    MiniBooNE particle identification
    "MiniBooNE_PID.txt"
    -> "MiniBooNE_PID.txt"
    50 attributes
    130065 instances
    binary class: signal/non-signal
    Preprocessing: file was dividing such that the first half was all from one class and the second half from the
    background
    Added explicit class labels
    changed sep to comma
    '''
    flat_data = np.loadtxt(get_data_path('MiniBooNE_PID.txt'),skiprows=1)
    num_signal, num_background = open(get_data_path('MiniBooNE_PID.txt')).readline().split()
    labels = np.concatenate([np.repeat('signal', num_signal),
                             np.repeat('background', num_background)])
    return Bunch(data=flat_data,
                 target=labels,
                 name='miniboone',
                 dataset_type='classification')
                 
def load_skin_segmentation():
    '''
    Skin Segmentation
    "Skin_NonSkin.txt"
    3 attributes
    245057 instances
    Binary class: skin(1), non-skin(2)
    Preprocesing
    ->
    "Skin_NonSkin.csv"
    changed sep to comma
    '''
    data = pd.read_csv(get_data_path('Skin_NonSkin.txt'), delim_whitespace=True, header=None)
    flat_data = data.ix[:, 0:2].values
    labels = data.ix[:, 3].values
    return Bunch(data=flat_data,
                 target=labels,
                 name='skin',
                 dataset_type='classification')
                 
    
def load_spambase():
    '''
    Spambase
    "spambase.data"
    4601 instances
    57 features
    binary class: spam(1), not(0)
    '''
    data = pd.read_csv(get_data_path('spambase.data'), header=None)
    flat_data = data.ix[:, 0:56]
    labels = data.ix[:, 57]
    return Bunch(data=flat_data.values,
                 target=labels.values,
                 name='spam',
                 dataset_type='classification')

def load_steel_plates_faults():
    '''
    Steel Plates Faults
    "faults.NNA"
    1941 instances
    27 attributes
    7 classes (types of faults)
    preprocessing -> "faults.csv"
    class was one hot encoded, changed to 0~6 single column code
    changed sep to comma
    '''
    data = np.loadtxt(get_data_path('faults.NNA'))
    features = data[:,0:27]
    labels = np.where(data[:, 27:])[1]
    return Bunch(data=features,
                 target=labels,
                 name='steel',
                 dataset_type='classification')
                 
def load_gas_sensor_array_drift():
    '''
    Gas Sensor Array Drift
    13910 instances
    128 attributes
    6 classes: 1: Ethanol; 2: Ethylene; 3:Ammonia; 4: Acetaldehyde; 5: Acetone; 6: Toluene
    Preprocessing->
    from libsvm format to normal csv format
    batch{1~10}.csv
    '''
    labels = []
    flat_data = []
    for i in range(1,11):
        with open('benchmark_datasets_original/batch%d.dat' % i) as f:
            for line in f:
                line_items = line.strip().split(' ')
                class_ = int(line_items[0])
                labels.append(class_)
                features = line_items[1:]
                features = [float(x.split(':')[1]) for x in features]
                flat_data.append(features)
    labels = np.array(labels)
    flat_data = np.array(flat_data)
    return Bunch(data=flat_data,
                 target=labels,
                 name='gas_sensor',
                 dataset_type='classification')
                 
def load_image_segmentation():
    '''
    Image Segmentation
    2310 instances
    19 features
    7 classes: brickface, sky, foliage, cement, window, path, grass.
    preprocessing: combine test traing into one set, remove header
    put class as last attribute
    {segmentation.data, segmentation.test} -> {segmentation.csv}
    '''
    train = pd.read_csv(get_data_path('segmentation.data'), header=None, skiprows=5)
    test = pd.read_csv(get_data_path('segmentation.test'), header=None, skiprows=5)
    data = pd.concat([train, test], axis=0)
    return Bunch(data=data.ix[:, 1:19].values,
                 target=data.ix[:, 0].values,
                 name='image_segmentation',
                 dataset_type='classification')
def load_landsat_satellite():
    '''
    Landsat Satellite
    6435 instances
    36 attributes
    6 classes:
    1	red soil 
    2	cotton crop 
    3	grey soil 
    4	damp grey soil 
    5	soil with vegetation stubble 
    (((6	mixture class (all types present) ))) not present
    7	very damp grey soil 
    {sat.trn, sat.tst} -> sat.csv
    use comma to sep,
    '''
    train = np.loadtxt(get_data_path('sat.trn'))
    test = np.loadtxt(get_data_path('sat.tst'))
    data = np.vstack([train, test])
    flat_data = data[:,:36]
    labels = data[:, -1]
    
    return Bunch(data=flat_data,
                 target=labels,
                 name='landsat',
                 dataset_type='classification')
    
def load_letter_recognition():
    '''
    Letter Recognition
    20000 instances
    16 attributes 
    26 classes (capital letters A-Z)
    moved classes to last attribute, letter-recognition.data -> letter-recognition.csv
    '''
    data = pd.read_csv(get_data_path('letter-recognition.data'), header=None)
    return Bunch(data=data.ix[:,1:16].values,
                 target=data.ix[:, 0].values,
                 name='letter-recognition',
                 dataset_type='classification')
                 
def load_handwritten_digits():
    '''
    Optical Recognition of Handwritten Digits
    5620 instances
    64 attributes
    10 classes (the digits 0-9)
    took the preprocessed version ,ptdigits.tes/optdigits.tra -> optdigits.csv
    '''
    train = np.loadtxt(get_data_path('optdigits.tra'), delimiter=',')
    test = np.loadtxt(get_data_path('optdigits.tes'), delimiter=',')
    data = np.vstack([train, test])
    
    return Bunch(data=data[:, :64],
                 target=data[:, -1],
                 name='handwritten-digits',
                 dataset_type='classification')
                 
def load_page_blocks():
    '''
    Page Blocks
    5473 instances
    10 attributes
    5 classes: text (1), horizontal line (2),
       picture (3), vertical line (4) and graphic (5).
    commas->page-blocks.csv
    '''
    data = np.loadtxt(get_data_path('page-blocks.data'))
    return Bunch(data=data[:, :10],
                 target=data[:, -1],
                 name='page-blocks',
                 dataset_type='classification')
    
def load_shuttle():
    '''
    Shuttle
    58000 instances
    9 attributes
    7 classes
    shuttle.trn, shuttle.tst -> shuttle.csv
    '''
    data = np.vstack([np.loadtxt(get_data_path('shuttle.trn')),
                      np.loadtxt(get_data_path('shuttle.tst'))])
    return Bunch(data=data[:, :9], target=data[:, -1], name='shuttle', dataset_type='classification')
    
def load_waveform():
    pass
def load_yeast():
    '''
    Yeast
    1484 instances
    1 sequence number + 8 real attributes
    10 classes (localization site of protein)
      CYT (cytosolic or cytoskeletal)                    463
      NUC (nuclear)                                      429
      MIT (mitochondrial)                                244
      ME3 (membrane protein, no N-terminal signal)       163
      ME2 (membrane protein, uncleaved signal)            51
      ME1 (membrane protein, cleaved signal)              44
      EXC (extracellular)                                 37
      VAC (vacuolar)                                      30
      POX (peroxisomal)                                   20
      ERL (endoplasmic reticulum lumen)                    5
    Note: first attribute(sequence number)IGNORED
    '''
    
    data = pd.read_csv(get_data_path('yeast.data'),delim_whitespace=True,header=None)
    flat_data = data.ix[:, 1:8].values
    labels = data.ix[:, 9].values
    return Bunch(data=flat_data, target=labels, name='yeast', dataset_type='classification')
def load_abalone():
    pass
def load_communities_and_crime():
    pass
def load_concrete_compressive_strength():
    pass
def load_wine():
    pass
def load_year_prediction():
    pass

def assert_correct_size(data, size_tup):
    try:
        assert data.data.shape == size_tup
        assert len(data.target) == size_tup[0]
    except AssertionError:
        raise Exception("%s Incorrect data size!" % data.name)

def assert_num_classes_correct(data, num_classes):
    try:
        assert len(np.unique(data.target)) == num_classes
    except AssertionError:
        raise Exception("%s Incorrrect number of classes!" % data.name)
        
if __name__=='__main__':
    print "Running some tests"
    magic = load_magic_gamma_telescope()
    assert magic.data.shape == (19020, 10)
    assert len(magic.target) == 19020
    assert magic.name == 'magic'
    
    miniboone = load_miniboone_particle_identification()
    assert miniboone.data.shape == (130064, 50)
    assert len(miniboone.target) == 130064
    assert miniboone.name == 'miniboone'
    
    dataset = load_skin_segmentation()
    assert_correct_size(dataset, (245057, 3))
    assert_num_classes_correct(dataset, 2)
    
    dataset = load_spambase()
    assert_correct_size(dataset, (4601, 57))
    assert_num_classes_correct(dataset, 2)
    
    dataset = load_steel_plates_faults()
    assert_correct_size(dataset, (1941, 27))
    assert_num_classes_correct(dataset, 7)
    
    
    gas_sensor = load_gas_sensor_array_drift()
    assert gas_sensor.data.shape == (13910, 128)
    assert len(gas_sensor.target) == 13910
    assert gas_sensor.name == 'gas_sensor'
    
    
    
    image_segmentation = load_image_segmentation()
    assert image_segmentation.data.shape == (2310, 19)
    assert len(image_segmentation.target) == 2310
    
    
    
    landsat = load_landsat_satellite()
    assert_correct_size(landsat, (6435, 36))
    assert_num_classes_correct(landsat, 6)
    
    letter = load_letter_recognition()
    assert_correct_size(letter, (20000, 16))
    assert_num_classes_correct(letter, 26)
    
    digits = load_handwritten_digits()
    assert_correct_size(digits, (5620, 64))
    assert_num_classes_correct(digits, 10)
    
    pb = load_page_blocks()
    assert_correct_size(pb, (5473, 10))
    assert_num_classes_correct(pb, 5)
    
    shuttle = load_shuttle()
    assert_correct_size(shuttle, (58000, 9))
    assert_num_classes_correct(shuttle, 7)
    
    yeast = load_yeast()
    assert_correct_size(yeast, (1484, 8))
    assert_num_classes_correct(yeast, 10)
    
    print "OK"