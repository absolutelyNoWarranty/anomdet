#java -jar elki.jar de.lmu.ifi.dbs.elki.application.greedyensemble.ComputeKNNOutlierScores -startk 3 -stepk 2 -maxk 30 -dbc.in aloi-27d-50000-max5-tot1508.csv.gz -app.out /tmp/aloi-results.ascii -algorithm.distancefunction colorhistogram.HistogramIntersectionDistanceFunction -db.index tree.spatial.rstarvariants.rstar.RStarTreeFactory -verbose 


#java -jar elki.jar de.lmu.ifi.dbs.elki.application.greedyensemble.GreedyEnsembleExperiment -dbc.in aloi-results.ascii -verbose

import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from anomdet.utils import normalize_scores

def elki_load_scores(filepath, replace_na_and_inf=True):
    scores_df = pd.read_csv(filepath, skiprows=1, index_col=0, sep=' ', header=None)
    
    rows = scores_df.iterrows()
    labels = rows.next()
    labels = labels[1].values.astype('float').astype('int')
        
    outlier_scores = scores_df.values[1:, :].astype('float').T.copy()
    #print "Scores loaded"
    if replace_na_and_inf:
        #print "Normalizing scores"
        outlier_scores = normalize_scores(outlier_scores)

        #Replace those stupid nas and infs
        #print "Replacing NA's and INF's with mean of scores"
        if replace_na_and_inf:
            for j in range(outlier_scores.shape[1]):
                scores = outlier_scores[:, j]
                bad = np.where(np.logical_or(np.isnan(scores), np.isinf(scores)))[0]
                outlier_scores[bad, j] = np.mean(np.ma.masked_invalid(scores))

    return (outlier_scores, labels)

def elki_knn_outliers(input_file, output_file=None, return_scores=True):
    if output_file is None:
        return_scores = True
        output_file = tempfile.mktemp()
    
    # Use DEVNULL to suppress output
    FNULL = open(os.devnull, 'w')    
    subprocess.call(['java', '-jar', '/home/alvin/bin/elki-bundle-0.6.5~20141030.jar',
                     'de.lmu.ifi.dbs.elki.application.greedyensemble.ComputeKNNOutlierScores',
                     '-startk', '3',
                     '-stepk', '2',
                     '-maxk', '30',
                     '-db.index', 'tree.spatial.rstarvariants.rstar.RStarTreeFactory',
                     '-dbc.in', input_file,
                     '-app.out', output_file,
                     ],
                     stdout=FNULL,
                     stderr=subprocess.STDOUT)

    if return_scores:
        return elki_load_scores(output_file)
        
def elki_greedy_ensemble(input_file, scaling='linear', return_raw_output=False):
    FNULL = open(os.devnull, 'w')
    
    scaling_dict = {'linear':'outlier.OutlierLinearScaling',
                    'gamma':'outlier.OutlierGammaScaling',
                    'standard_deviation':'outlier.StandardDeviationScaling',
                    'sigmoid':'outlier.SigmoidOutlierScalingFunction'}
    scaling = scaling_dict[scaling]
    output = subprocess.check_output(['java', '-jar', '/home/alvin/bin/elki-bundle-0.6.5~20141030.jar',
                     'de.lmu.ifi.dbs.elki.application.greedyensemble.GreedyEnsembleExperiment',
                     '-ensemble.measure', 'PEARSON',
                     '-ensemble.scaling', scaling,
                     '-dbc.in', input_file,
                     '-verbose'],
                     stderr=FNULL)
                     
    if return_raw_output: return output
    
    tmp_ind = output.find('Naive ensemble AUC')
    naive_ensemble_auc = float(output[tmp_ind + len('Naive ensemble AUC:'):].strip().split(' ')[0])
    
    tmp_ind = output.find('Greedy ensemble AUC:')
    greedy_ensemble_auc = float(output[tmp_ind + len('Greedy ensemble AUC:'):].strip().split(' ')[0])
    
    return (naive_ensemble_auc, greedy_ensemble_auc)
    
