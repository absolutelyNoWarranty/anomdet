import numpy as np

def imatestfunction():
    pass
    
#UTILS FOR SAVING AND LOADING NP.MATRICES CERATED WITH A RANDOMSEED

def save_stuff(thingy, desc, random_seed):
    if not isinstance(desc, str):
        desc = '-'.join(desc)
    desc = desc.replace('/','_')
    filename = desc + "_rs" + str(random_seed) + '.npy'
    print "Saving: ", filename
    np.save(filename, thingy)
    
def load_stuff(desc, random_seed):
    if not isinstance(desc, str):
        desc = '-'.join(desc)
    desc = desc.replace('/','_')
    filename = desc + "_rs" + str(random_seed) + '.npy'
    print "Trying to load:", filename
    try:
        return np.load(filename)
    except IOError:
        print filename, "not found"
        return None
