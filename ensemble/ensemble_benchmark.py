import numpy as np

from ..utils import normalize_scores
from ..utils import maybe_default_random_state

def benchmark_ensemble(datasets=None, methods=None, one_at_a_time=False, combine=None, random_state=None):
    '''
    dat : list of Datasets
        OR list of (Dataset, subsampling_ratio, n_iter)
    
    one_at_a_time : boolean
        If true, then run datasets one outlier at a time
    methods : list of methods
    combine : list of combine function to combine score matrix
    
    Returns
    -------
    rows
    '''
    
    rs = maybe_default_random_state(random_state)
    
    n_datasets = len(datasets)
    n_methods = len(methods)
    n_combine_methods = len(combine)
    
    rows = []
    vals = []
    for item in datasets:
        row = []
        
        
        if isinstance(item, tuple):
            Dat, subsampling_ratio, n_iter = item
            aucs = np.empty((n_iter, n_combine_methods+1)) #combine methods + expected auc
            for i in range(n_iter):
                row = []
                dat = Dat.get_subsampled(anomaly_ratio=subsampling_ratio, random_state=rs.randint(0, 2**16))
                
                scores = np.empty((dat.X.shape[0], n_methods))
                avg_auc = 0.
                for j, clf in enumerate(methods):
                    preds = clf.fit(dat.X).predict(dat.X)
                    scores[:, j] = preds
                    avg_auc += dat.evaluate(preds)
                avg_auc /= n_methods
                
                row.append(avg_auc)
                for comb in combine:
                    preds = comb(scores)
                    row.append(dat.evaluate(preds))
                aucs[i] = row
            #aucs.mean(axis=0)
            #aucs.std(axis=0)
            rows.append([Dat.name] +["{:.2f}+/-{:.2f}".format(mu, sigma) for (mu, sigma) in zip(aucs.mean(axis=0), aucs.std(axis=0))])
            vals.append(aucs)
        elif one_at_a_time:
            Dat = item
            
            n_iter = sum(Dat.y)
            aucs = np.empty((n_iter, n_combine_methods+1)) #combine methods + expected auc
            
            #for i in range(n_iter):
            for i, dat in enumerate(Dat.iter_individual()):
                row = []
                #dat = Dat.get_subsampled(anomaly_ratio=subsampling_ratio, random_state=rs.randint(0, 2**16))
                
                scores = np.empty((dat.X.shape[0], n_methods))
                avg_auc = 0.
                for j, clf in enumerate(methods):
                    preds = clf.fit(dat.X).predict(dat.X)
                    scores[:, j] = preds
                    avg_auc += dat.evaluate(preds)
                avg_auc /= n_methods
                
                row.append(avg_auc)
                for comb in combine:
                    preds = comb(scores)
                    row.append(dat.evaluate(preds))
                aucs[i] = row
            #aucs.mean(axis=0)
            #aucs.std(axis=0)
            rows.append([Dat.name] + ["{:.2f}+/-{:.2f}".format(mu, sigma) for (mu, sigma) in zip(aucs.mean(axis=0), aucs.std(axis=0))])
        else:
            dat = item
            row.append(dat.name)
        
            scores = np.empty((dat.X.shape[0], n_methods))
            avg_auc = 0.
            for j, clf in enumerate(methods):
                preds = clf.fit(dat.X).predict(dat.X)
                scores[:, j] = preds
                avg_auc += dat.evaluate(preds)
            avg_auc /= n_methods
            
            row.append(avg_auc)
            for comb in combine:
                preds = comb(scores)
                row.append(dat.evaluate(preds))
            
            
            
            
            rows.append(row)
    
    return rows, vals