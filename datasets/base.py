from ..utils.dataset_info import DatasetInfo
from sklearn.metrics import roc_auc_score
from ..utils.simple_timer import SimpleTimer
from ..utils import unique_rows

class OutlierDataset(object):
    def __init__(self, X, y=None, name=None, pos_class_name="outlier",
                 neg_class_name="normal", allow_duplicates=True):
        self._X = X
        self._y = y
        self.name = name
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
    
    def benchmark(self, thingy):
        '''
        Given a `thingy`, benchmark it on this dataset by calling the thingy's
        predict.
        '''
        self.timer.tic()
        preds = thingy.fit(self.data).predict(self.data)
        self.timer.toc()
        print self.timer
        auc = roc_auc_score(self.labels, preds)
        print auc
        return auc
        
    def evaluate(self, preds):
        '''
        Evaluate `preds` on this OutlierDataset using the AUC of the ROC curve.
        '''
        return roc_auc_score(self.labels, preds)
    
    @property
    def info(self):
        return self._info.calc(self.y)
    
    def __str__(self):
        return self.info.__str__()
        
        