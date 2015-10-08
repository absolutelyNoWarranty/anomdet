# Download and split the DARPA data (KDDCUP1999) by attack type into individual csv files
import os
import urllib
import gzip
from collections import defaultdict

import pandas as pd

from sklearn.metrics import roc_auc_score

from anomdet.utils import replace_invalid_scores

THIS_DIR = os.path.dirname(os.path.abspath(__file__)) # absolute path to directory this source file is in

column_names = ["duration",
"protocol_type",
"service",
"flag",
"src_bytes",
"dst_bytes",
"land",
"wrong_fragment",
"urgent",
"hot",
"num_failed_logins",
"logged_in",
"num_compromised",
"root_shell",
"su_attempted",
"num_root",
"num_file_creations",
"num_shells",
"num_access_files",
"num_outbound_cmds",
"is_host_login",
"is_guest_login",
"count",
"srv_count",
"serror_rate",
"srv_serror_rate",
"rerror_rate",
"srv_rerror_rate",
"same_srv_rate",
"diff_srv_rate",
"srv_diff_host_rate",
"dst_host_count",
"dst_host_srv_count",
"dst_host_same_srv_rate",
"dst_host_diff_srv_rate",
"dst_host_same_src_port_rate",
"dst_host_srv_diff_host_rate",
"dst_host_serror_rate",
"dst_host_srv_serror_rate",
"dst_host_rerror_rate",
"dst_host_srv_rerror_rate",
"attack_label"]

def download_and_split(filedir=THIS_DIR):
    '''
    Downloads the DARPA data if not found and splits by attack type.
    1. Only tcp data is used.
    2. 38 binary and continuous features are used (starting from 0, columns 1, 2, 3 are removed)
    '''
    filepath = os.path.join(filedir, "kddcup.data_10_percent.gz")
    if os.path.exists(filepath):
        print "Found kddcup.data_10_percent.gz!"
    else:
        URL = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
        
        print "Downloading kddcup.data_10_percent.gz from\n%s\nPlease wait ..." % URL 
        urllib.urlretrieve(URL, os.path.join(filedir, "kddcup.data_10_percent.gz"))
        
    print "Splitting data ..."
    f = gzip.open(filepath, "r")
    X = pd.read_csv(f, names=column_names, header=None)
    f.close()
    X_tcp = X.loc[X.protocol_type == 'tcp']

    attack_labels = X_tcp.attack_label.unique().tolist()
    X_by_attack_type =  {k : X_tcp.loc[X_tcp.attack_label == k] for k in attack_labels}

    for attack in attack_labels:
        if attack == 'normal.':
            continue
        df = pd.concat([X_by_attack_type['normal.'], X_by_attack_type[attack]])
        # remove string columns 1, 2, and 3
        df = df.iloc[:, [0]+range(4, df.shape[1])]
        df.to_csv(os.path.join(filedir, "kddcup10percent-tcp-attack_%s.csv"% attack.replace('.', '')), index=False)
    
    print "Done!"

def benchmark_darpa(models, attack_types='all', metric=None):
    '''
    atack_types : list or str
        list of attack types to benchmark, can be:
            back dos
            buffer_overflow u2r
            ftp_write r2l
            guess_passwd r2l
            imap r2l
            ipsweep probe
            land dos
            loadmodule u2r
            multihop r2l
            >>>neptune dos # too many
            >>>nmap probe # not available
            perl u2r
            phf r2l
            >>>>pod dos # not available
            portsweep probe
            rootkit u2r
            satan probe
            >>>smurf dos # not available
            spy r2l
            >>>>teardrop dos #not available
            warezclient r2l
            warezmaster r2l
            
            or one of [dos, u2r, r2l, probe] which will use all attacks in that category
    '''
    
    attacks_categories = '''back dos
                            buffer_overflow u2r
                            ftp_write r2l
                            guess_passwd r2l
                            imap r2l
                            ipsweep probe
                            land dos
                            loadmodule u2r
                            multihop r2l
                            perl u2r
                            phf r2l
                            portsweep probe
                            rootkit u2r
                            satan probe
                            spy r2l
                            warezclient r2l
                            warezmaster r2l'''
                            
    if metric is None:
        metric = roc_auc_score
                            
    d = defaultdict(list)
    for item in attacks_categories.split('\n'):
        attack, cat = item.strip().split()
        d[cat].append(attack)
    attacks = reduce(lambda a,b: a+b, d.values())
    
    if attack_types == "all":
        attacks_to_load = attacks
    elif attack_types in ["dos", "u2r", "r2l", "probe"]:
        attacks_to_load = d[attack_types]
    elif attack_types in attacks:
        attacks_to_load = [attack_types]
    else:
        raise Exception("Not a valid attack type")
    
    perf_scores = defaultdict(list)
    for name, model in models.iteritems():
        for attack in attacks_to_load:
            X = pd.read_csv(os.path.join(THIS_DIR, "kddcup10percent-tcp-attack_%s.csv" % attack))
            y = (X.iloc[:, -1] != "normal.").values
            X = X.iloc[:, :-1].values
            model.fit(X, None)
            preds = model.predict(X)
            preds = replace_invalid_scores(preds)
            perf_scores[name].append(metric(y, preds))
    
    return perf_scores
            
if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        download_and_split(sys.argv[1])
    else:
        download_and_split()
