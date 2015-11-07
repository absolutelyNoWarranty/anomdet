from collections import namedtuple

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from ..utils.dataset_info import DatasetInfo
from ..utils.simple_timer import SimpleTimer
from ..utils import unique_rows, maybe_default_random_state

from .utils import get_subsample_indices

def iter_sampled_outliers(self, outlier_ratio=0.05, random_state=None, overlap=True):
    random_state = maybe_default_random_state(random_state)
    num_pos_sample = int(np.ceil(outlier_ratio * len(self.y)))
    
    # shuffle the pos class indices
    pos_y = np.where(self.y)[0]
    shuffle_ind = np.argsort(random_state.rand(len(pos_y)))
    neg_y = np.where(np.logical_not(self.y))[0]
    
    if overlap:
        for i in range(0, len(pos_y)):
            if i + num_pos_sample <= len(pos_y):
                subset_ind = np.append(neg_y, pos_y[shuffle_ind][i: i + num_pos_sample])
            else:
                subset_ind = np.append(np.append(neg_y, pos_y[shuffle_ind][i:]),
                             pos_y[shuffle_ind][0:(i+num_pos_sample)-len(pos_y)])
                
            yield OutlierDataset(self.X[subset_ind], self.y[subset_ind],
                                  name=self.name+" subsampled",
                                  pos_class_name=self.pos_class_name,
                                  neg_class_name=self.neg_class_name)
    else:
        for i in range(0, len(pos_y), num_pos_sample):
            if i + num_pos_sample <= len(pos_y):
                subset_ind = np.append(neg_y, pos_y[shuffle_ind][i: i + num_pos_sample])
            else:
                subset_ind = np.append(np.append(neg_y, pos_y[shuffle_ind][i:]),
                             pos_y[shuffle_ind][0:(i+num_pos_sample)-len(pos_y)])

            yield OutlierDataset(self.X[subset_ind], self.y[subset_ind],
                                  name=self.name+" subsampled",
                                  pos_class_name=self.pos_class_name,
                                  neg_class_name=self.neg_class_name)
            
            
    
class OutlierDataset(object):
    def __init__(self, X, y=None, name=None, pos_class_name="outlier",
                 neg_class_name="normal", allow_duplicates=True):
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of rows in X is not the same as length of "
                             "y.")
            
        self._X = X
        self._y = y
        self.name = name
        self.pos_class_name = pos_class_name
        self.neg_class_name = neg_class_name
        
        self._info = DatasetInfo(dataset_name=self.name,
                                pos_class_name=pos_class_name,
                                neg_class_name=neg_class_name)
        self.timer = SimpleTimer()
        
        self.set_duplicates_allowed(allow_duplicates)
        
    @property
    def X(self):
        if self.duplicates_allowed:
            return self._X
        else:
            return self._X[self.unique_ind]
            
    @property
    def data(self):
        return self.X
    
    @property
    def y(self):
        if self.duplicates_allowed:
            return self._y
        else:
            return self._y[self.unique_ind]
    
    @property
    def labels(self):
        return self.y
    
    def set_duplicates_allowed(self, is_ok=True):
        self.duplicates_allowed = is_ok
        if not is_ok:
            _, self.unique_ind = unique_rows(self._X, return_index=True)
    
    def benchmark(self, thingy, k=None, threshold=None, verbose=False):
        '''
        Given a `thingy`, benchmark it on this dataset by calling the thingy's
        predict.
        
        Parameters
        ----------
        thingy : something that can predict outlier scores
        
        k : float or int, optional, (default=None)
            The top k objects to consider when sorted by outlier scores

        threshold : float or int, optional, (default=None)
            The threshold for outlier scores above which an item is considered
            an outlier
        
        verbose : bool, (default=False)
            Whether to print stuff while running.
        
        Returns
        -------
        If k and threshold are both None:
        auc : The auc
        
        Or, if either k OR threshold is provided        
        result : A named tuple of the auc, precision, recall, and f1_score
                 where precision, recall, and f1_score are calculated assuming
                 objects which are in the top-k OR are above the threshold are
                 outliers (positive) and others are not (negative).
            
        '''
        
        evaluation_funcs = dict(roc_auc=roc_auc_score,
                                precision=precision_score,
                                recall=recall_score,
                                f1=f1_score)
                                
        if verbose: self.timer.tic()
        outlier_scores = thingy.fit(self.data).predict(self.data)
        if verbose: self.timer.toc(); print self.timer
        auc = self.evaluate(outlier_scores)
        
        if k is None and threshold is None:
            if verbose: print auc
            return auc
            
        elif k is not None and threshold is not None:
            raise ValueError("k and threshold both set")
        elif k is not None:
            top_ind = np.argsort(outlier_scores)[-k:]
        else:
            top_ind = np.where(outlier_scores >= threshold)[0]

        preds = np.zeros_like(outlier_scores)
        preds[top_ind] = 1
        
        MetricsBundle = namedtuple('Metrics', ['auc', 'precision', 'recall', 'f1'])
        result = MetricsBundle(auc=auc,
                               precision=precision_score(self.y, preds),
                               recall=recall_score(self.y, preds),
                               f1=f1_score(self.y, preds))
                               
        if verbose: print result
        return result
        
    def evaluate(self, preds):
        '''
        Evaluate `preds` on this OutlierDataset using the AUC of the ROC curve.
        '''
        return roc_auc_score(self.labels, preds)
    
    @property
    def info(self):
        return self._info.calc(self.y)
    
    def __str__(self):
        return self.name
   
        
dumdum = OutlierDataset(np.array([[1., 0.5],
                                  [2., 0.6],
                                  [3., 0.001],
                                  [4., -1.5],
                                  [5., -2],
                                  [6., 0.00],
                                  [0.001, 1.],
                                  [0.001, 2.],
                                  [0.001, 3.],
                                  [0.001, 4.],
                                  [0.001, 5.],
                                  [0.001, 6.],
                                  [0.001, 7.],
                                  [0.001, 8.],
                                  [0.001, 9.],
                                  [0.001, 10.],
                                  [0.001, 11.],
                                  [0.001, 12.],
                                  [0.001, 13.],
                                  [0.001, 14.]]),
                        np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                 dtype=bool),
                        name="a little dummy")