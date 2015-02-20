import numpy as np

def bincount2(labels):
    '''
    Wrapper for numpy's bincount so that it works for counting labels (which are
    non-numeric)
    '''
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(labels)
    return zip(label_enc.classes_, np.bincount(label_enc.transform(labels)))

def var(P):
    '''
    The definition of variance applied to multidimensional points
    (i.e. squared difference changed to norm of difference)
    '''
    n = P.shape[0]
    mu = np.mean(P, axis=0)
    return np.sqrt(np.sum((P - mu).T.dot(P-mu)) / n)
    
def minimum_spanning_tree(X, copy_X=True):
    '''
    Given that `X` are edge weights of a fully connected graph
    returns the minimum spanning tree
    '''
    if copy_X:
        X = X.copy()
 
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
     
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
    
    color1_vertices = set(visited_vertices)
    color2_vertices = set() 
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                      
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        
        if new_edge[0] in color1_vertices:
            color2_vertices.add(new_edge[1])
        else:
            color1_vertices.add(new_edge[1])
        
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
    return np.vstack(spanning_edges), color1_vertices, color2_vertices
    
def subsample(dataset, N=10000):
    '''
    subsamples dataset so that it has N instances
    stratified
    '''
    
    dataset = copy.copy(dataset)
    
    # binc = bincount2(dataset.target)
    # downsample_ratio = float(N) / len(dataset.target)
    # data = []
    # target = []
    # for class_, count in binc:
        # rs = StratifiedShuffleSplit(count, n_iter=1, train_size=downsample_ratio)
        # indices = tuple(rs)[0][0]
        # data.append(dataset.data[np.where(dataset.target == class_)[0][indices], :])
        # target.append(np.repeat(class_, len(indices)))

    indices = tuple(StratifiedShuffleSplit(dataset.target, n_iter=1, train_size=N))[0][0]
    
    
    
    #data = np.vstack(data)
    #target = np.concatenate(target)
    
    data = dataset.data[indices, :]
    target = dataset.target[indices]
    
    dataset.data = data
    dataset.target = target
    return dataset

def visualize_anomaly_rankings(ax, y_true, y_score, **kwargs):
    """
    A helper function to visualize how well the data was scored for anomalies

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    y_true : array
       The true labels

    y_score : array
       The target scores or probabilities

    pos_label : int
    Returns
    -------
    out : list
        list of artists added
    """
    y_sorted_by_preds = y_true[np.argsort(y_score)]
    
    n = len(y_true)
    n_p = np.sum(y_true)
    
    
    out = ax.plot(np.repeat(0, n_p), np.where(y_sorted_by_preds)[0], 'o', markersize=5, fillstyle='none')
    out += ax.plot(np.repeat(0, n-n_p), np.where(np.logical_not(y_sorted_by_preds))[0], ',')

    ax.get_xaxis().set_visible(False)
    ax.set_yticklabels([])

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0,n+0.5)
    #ax.set_aspect(1.1)
    
    return out
    
def top_n_precision(y, ranking, n):
    return np.mean(y[np.argsort(ranking)[-n:]])

def precision_at_k(y, rankings, k=None):
    num_positives = np.sum(y)
    if not k:
        k = num_positives  # number of outliers
    true_positives_for_top_k = np.sum(y[np.argsort(rankings)][-k:])
    return true_positives_for_top_k / float(num_positives)