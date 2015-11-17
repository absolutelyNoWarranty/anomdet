import numpy as np
from .tabulate import tabulate

class DatasetInfo(object):
    '''
    Expects to be given labels that are binary 1/0 or True/False
    
    pos_class_name : str, name for the 1 or True class
    neg_class_name : str, name for the 0 or False class
    
    '''
    
    def __init__(self, dataset_name="Some Dataset", pos_class_name="outlier",
                 neg_class_name="normal"):
        self.dataset_name = dataset_name
        self.pos_class_name = pos_class_name
        self.neg_class_name = neg_class_name
  
    def calc(self, labels):
        labels = np.asarray(labels, dtype=bool)
        
        num_1 = np.sum(labels)
        num_0 = np.sum(np.logical_not(labels))
        total = len(labels)
     
        assert num_1 + num_0 == total

        self.num_1 = num_1
        self.num_0 = num_0
        self.total = total
        
        return self
    
    def __call__(self, *args):
        return self.calc(*args)
    
    def table(self, tablefmt="pipe"):
        header = [self.dataset_name, self.pos_class_name, self.neg_class_name]
        rows = [["Number", self.num_1, self.num_0],
                ["Perct.", "{:2f}".format(self.num_1 * 100.0/self.total),
                 "{:2f}".format(self.num_0 * 100.0/self.total)]]
        return tabulate(rows, headers=header, tablefmt=tablefmt)
    
    def __str__(self):
        return "\n".join([
            "'{name}' Info:".format(name=self.dataset_name),
            
            "# of {pos_class} = {number}".format(
                pos_class=self.pos_class_name,
                number=self.num_1),
                
            "# of {neg_class} = {number}".format(
                neg_class=self.neg_class_name,
                number=self.num_0),
        
            "{percent:2f}% of the data are {pos_class}(s)".format(
                percent=self.num_1 * 100.0 / self.total,
                pos_class=self.pos_class_name),

            "Ratio of {pos_class} to {neg_class}".format(
                pos_class=self.pos_class_name,
                neg_class=self.neg_class_name),
            
            "[{pos_class}]:[{neg_class}] = 1.0 : {num_neg_per_pos:1f}".format(
                pos_class=self.pos_class_name,
                neg_class=self.neg_class_name,
                num_neg_per_pos=self.num_0 * 1.0 / self.num_1)])
            
my_dataset_info = DatasetInfo()