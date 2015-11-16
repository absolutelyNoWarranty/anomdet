from collections import OrderedDict
import numpy as np
from .base import OutlierDataset
from .utils import get_subsample_indices
from ..utils import maybe_default_random_state
from ..utils.tabulate import tabulate

class Dataset(object):
    '''
    Parameters
    ----------
    X : arary-like, shape (n_samples, n_features)
        the data
    y : array-like, shape (n_samples)
        the labels
    classes_ : dict, optional (default=None)
        A dictionary of (k=class representation in y, and v=class name)
        
    Attributes
    ----------
    X : the data
    y : the class labels
    name : the name of this Dataset
    classes_ : the class names, optional
    features : the feature names, optional
    header: same as feature names
    
    num_classes : derived feature from len(classes_)
    '''
    def __init__(self, X, y, name=None, classes_=None, features=None,
                 show_instance_f=None):
        self.X = X
        self.y = y
        
        self.name = name
        
        if classes_ is None:
            unique_in_y = np.unique(y)
            classes_ = dict(zip(unique_in_y, unique_in_y))
        try:
            self.classes_ = OrderedDict(sorted(dict(classes_).iteritems()))
        except TypeError:
            raise TypeError("Cannot convert classes_ to a dict")

        if features is None:
            features = ["X{0}".format(i) for i in xrange(X.shape[1])]
        self.features = features
        
        self._show_instance_f = show_instance_f
    
    @property
    def num_classes(self):
        return len(self.classes_)
    
    def instance_repr(self, i):
        '''
        Get the representation of i-th instance.
        '''
        if self._show_instance_f is None:
            # Use tabulate
            return tabulate([np.append(self.X[i], self.y[i])],
                            tablefmt="pipe",
                            headers=self.features + ["Label"],
                            floatfmt=".2")
        
        else:
            self._show_instance_f(self.X[i:(i+1)])
        
    def show_instance(self, i):
        '''
        Helper function to display an instance.
        '''
        if self._show_instance_f is None:
            print self.instance_repr(i)
            
        else:
            pass
        
    
    def _to_class_repr(self, names):
        '''
        Given an iterable of class names,
        returns the class reprensentations in y
        '''
        if not hasattr(names, '__iter__'):
            name = names
            if name in self.classes_: # already in y repr form
                return name
            
            search_v = [k for (k, v) in self.classes_.iteritems() if v == name]
            if search_v:
                return search_v[0]
                
            raise ValueError("Unknown class.")
            
        return [self._to_class_repr(name) for name in names]
    
    def as_outlier_dataset(self, normal_classes=None, outlier_classes=None):
        '''
        Return this dataset as an OutlierDataset
        
        Parameters
        ----------
        normal_classes
        outlier_classes
        '''
      
        if normal_classes is not None:
            if outlier_classes is not None:
                raise ValueError("Both normal and outlier classes are given?")
            normal_classes = self._to_class_repr(normal_classes)
            outlier_classes = list(set(self.classes_.keys()).difference(set(normal_classes)))
        else:
            if outlier_classes is None:
                raise ValueError("Please provide normal or outlier classes")
            outlier_classes = self._to_class_repr(outlier_classes)

        y = np.in1d(self.y, outlier_classes) 
        return OutlierDataset(self.X, y)
    
    def create_outlier_dataset(self, n_to_select=None, normal_classes=None,
                               outlier_classes=None, replace=False,
                               random_state=None):
        '''
        Create an OutlierDataset from this Dataset
        
        Parameters
        ----------
        `n_to_select` : array-like, shape(`classes`, ) 
            The number to select from each class.
            Elements can be:
                int, the exact number
                float, percentage
                "all", all
                
        `replace` : boolean, optional (default=False)
            Whether to sample with replacement or without.
        '''
        rs = maybe_default_random_state(random_state)
        init_state = rs.get_state()
        
        if n_to_select is None:
            init_state = rs.get_state()
            
            print "Creating a random OutlierDataset!"
            normal_class_ind = rs.choice(self.num_classes, 1)[0]
            
            n_to_select = [1] * self.num_classes
            n_to_select[normal_class_ind] = 'all'
            normal_classes = [self.classes_.keys()[normal_class_ind]]
            
            rs.set_state(init_state)
            
        ind = get_subsample_indices(self.classes_.keys(), self.y, n_to_select,
                                    replace=False, random_state=rs)
        
        X_ = self.X[ind]
        y_ = self.y[ind]
        if normal_classes is not None:
            if outlier_classes is not None:
                raise ValueError("Both normal and outlier classes are given?")
            normal_classes = self._to_class_repr(normal_classes)
            outlier_classes = list(set(self.classes_.keys()).difference(set(normal_classes)))
        else:
            if outlier_classes is None:
                raise ValueError("Please provide normal or outlier classes")
            outlier_classes = self._to_class_repr(outlier_classes)

        y_ = np.in1d(y_, outlier_classes) 
        return OutlierDataset(X_, y_)
    
    