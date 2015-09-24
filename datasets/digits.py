import os
from collections import namedtuple, defaultdict, OrderedDict
import numpy as np
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from PIL import Image
from sklearn.datasets.base import Bunch

_digits_X, _digits_y = None, None

def _create_digits_kaggle_memmap():
    print "Creating memmap file."
    filepath1 = os.path.join(os.path.dirname(__file__), 'train.csv')
    filepath2 = os.path.join(os.path.dirname(__file__), 'digits_train.dat')
    data = np.loadtxt(filepath1, delimiter=',', skiprows=1)
    memmapped_data = np.memmap(filepath2, dtype=np.float64, shape=(42000, 785), mode='w+')
    memmapped_data[:] = data
    del memmapped_data

def load_digits_kaggle(filepath=None, cache=True, use_memmap=True):
    if not use_memmap:
    
        global _digits_X
        global _digits_y
        
        if _digits_X is not None:
            return _digits_X, _digits_y
        
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'train.csv')
        
        X = np.loadtxt(filepath, delimiter=',', skiprows=1)
        y = X[:, 0].astype(int)
        X = X[:, 1:]
        
        if cache:
            _digits_X = X
            _digits_y = y
        
    else:
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'digits_train.dat')
        data = np.memmap(filepath, dtype=np.float64, shape=(42000,785), mode='c')
        X = data[:, 1:]
        y = data[:, 0].astype(int)
        del data
    return (X, y)

def get_subsample_indices(classes, y, select, replace=False, random_state=None):
    '''
    Returns indices which can be used to subsample a dataset X.
    Input:
        `y` - the class labels of X
        `select` - the number to select from each class
        `replace` - whether to sample with replacement or without
            default: False
    '''
    
    random_state = check_random_state(random_state)
    
    result = []
    #classes = np.unique(y)
    for num_to_select, class_ in zip(select, classes):
        indices = np.where(y==class_)[0]
        result.append(random_state.choice(indices, num_to_select, replace=replace))
    result = np.concatenate(result)
    
    # Randomize the order of the indices to prevent strange things from happening
    n = len(result)
    result = result[random_state.choice(n, n, replace=False)]
    
    return result
    
def load_digits_data(n_samples_per_class=None, random_state=None):
    '''
    Returns digits data.
    Input:
        `n_samples_per_class` : list or int or dict
            a list of the number of samples to return per class, 
            or a dict specifying the number of samples for each digit (those not included are 0)
            or, if a int, returns that number of samples for each class
    Output: Data X and labels y
    '''
    
    random_state = check_random_state(random_state)
    
    if n_samples_per_class is None:
        n_samples_per_class = [1] * 10
    
    if isinstance(n_samples_per_class, int):
        n_samples_per_class = [n_samples_per_class] * 10
    
    if isinstance(n_samples_per_class, dict):
        tmp = []
        for i in range(10):
            tmp.append(n_samples_per_class.get(i, 0))
        n_samples_per_class = tmp
        
    all_X, all_y = load_digits_kaggle()
    
    ind = get_subsample_indices(range(10), all_y, select=n_samples_per_class, random_state=random_state)
            
    X = all_X[ind]
    y = all_y[ind]
    
    return namedtuple('Dataset', ['X', 'y'])(X, y)


def benchmark_digits(models, metric=None, n_iter=1, num_train_samples=1000, num_test_samples=1000, anomaly_ratio=0.05, random_state=None, single_digit_as_normal=True, mode="train_test", return_data=False):
    '''
    mode : 1, or "train_test", "dirty_train" - training set has anomalies
    2, or "one_class" or "clean_train" - training set has no anomalies
    3, "unsupervised" - no training set
    
    0, generate and return data for the first iteration, don't run anything
    '''

    if models==None or len(models)==0:
        models = dict()
        return_data=True
    
    if mode == 1 or mode == "train_test" or mode == "dirty_train":
        mode = 1
    elif mode == 2 or mode == "one_class" or mode == "clean_train":
        mode = 2
    elif mode == 3 or mode == "unsupervised":
        mode = 3
    else:
        raise Exception("Missing mode")
    
    if metric is None:
        metric = roc_auc_score
        
    random_state = check_random_state(random_state)
    
    #Random states for selecting train/test datasets
    rs_train = check_random_state(random_state.randint(1000))
    rs_test = check_random_state(random_state.randint(1000))
    
    for _, model in models.iteritems():
        seed = random_state.randint(1000)
        try:  # Not all estimator accept a random_state
            #model.set_params(random_state=seed) ##TODO
            model.random_state = seed
        #except ValueError:
        except AttributeError:
            pass
    
    
    if mode == 0:
        n_iter = 1
        n_train_normal = int(num_train_samples * (1.0 - anomaly_ratio))
        n_train_anomaly = num_train_samples - n_train_normal
    elif mode == 1:
        n_train_normal = int(num_train_samples * (1.0 - anomaly_ratio))
        n_train_anomaly = num_train_samples - n_train_normal
    elif mode == 2:
        n_train_normal = num_train_samples
        n_train_anomaly = 0
        
    n_test_normal = int(num_test_samples * (1.0 - anomaly_ratio))
    n_test_anomaly = num_test_samples - n_test_normal
    
    all_X, all_y = load_digits_kaggle()
    
    perf_scores = defaultdict(list)
    if return_data: # Collect train and test data and return
        train_data = []
        test_data = []
    all_preds = defaultdict(list)
    for experiment_iter in range(n_iter):
        for i in range(10):
            if single_digit_as_normal: # digit i is the normal class
                normal_class = i
            
                #binary_y
                bin_y = all_y != normal_class; classes = [True, False]
                
                if mode != 3:
                    train_ind = get_subsample_indices(classes, bin_y, select=[n_train_anomaly, n_train_normal], random_state=rs_train)
                
                test_ind = get_subsample_indices(classes, bin_y, select=[n_test_anomaly, n_test_normal], random_state=rs_test)
            
            else: # digit i is the anomaly class, stratify sample from other classes to get normal class
                anomaly_class = i
                if mode!=3:
                    select = [int(n_train_normal/9)] * 10
                    for i_ in range(n_train_normal%9):
                        if i == i_:
                            select[-1] += 1
                        select[i_] += 1
                        
                    select[i] = n_train_anomaly
                    train_ind = get_subsample_indices(range(10), all_y, select=select, random_state=rs_train)
                #import pdb;pdb.set_trace()
                select = [int(n_test_normal/9)] * 10
                for i_ in range(n_test_normal%9):
                    if i == i_:
                        select[-1] += 1
                    select[i_] += 1
                    
                select[i] = n_test_anomaly
                test_ind = get_subsample_indices(range(10), all_y, select=select, random_state=rs_test)
            
                #binary_y
                bin_y = all_y == anomaly_class

            if mode != 3:
                X_train = all_X[train_ind]
                y_train = bin_y[train_ind]
                assert len(y_train) == num_train_samples
                assert sum(y_train) == n_train_anomaly

            X_test = all_X[test_ind]
            y_test = bin_y[test_ind]
        
            
            assert len(y_test) == num_test_samples
            assert sum(y_test) == n_test_anomaly
        
            if return_data:
                if mode != 3:
                    train_data.append((X_train, y_train))
                test_data.append((X_test, y_test))
                
            for name, model in models.iteritems():
                if mode != 3:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_test, y=None)
                preds = model.predict(X_test)
                perf_scores[name].append(metric(y_test, preds))
                all_preds[name].append(preds)

    avg_perf_scores = OrderedDict()
    for name in models.iterkeys():
        avg_perf_scores[name] = np.array(perf_scores[name]).reshape(n_iter, 10).mean(axis=0)

    ret_res = Bunch()
    
    if mode != 0 and models:
        ret_res['performance_scores'] = perf_scores
        ret_res['avg_perf_scores'] = avg_perf_scores
        ret_res['all_preds'] = all_preds
    
    if return_data:
        if mode != 3:
            ret_res['train_data'] = train_data
        else:
            ret_res['train_data'] = None
        ret_res['test_data'] = test_data
                    
    return ret_res
    
def benchmark_digits1(model, metric, n_iter=1, num_train_samples=1000, num_test_samples=1000, anomaly_ratio=0.05, random_state=None):
    
    random_state = check_random_state(random_state)
    
    model.random_state = check_random_state(random_state.randint(1000))
    
    n_train_normal = int(num_train_samples * (1.0 - anomaly_ratio))
    n_train_anomaly = num_train_samples - n_train_normal
    
    n_test_normal = int(num_test_samples * (1.0 - anomaly_ratio))
    n_test_anomaly = num_test_samples - n_test_normal
    
    all_X, all_y = load_digits_kaggle()
    
    perf_scores = []
    train_data = []
    test_data = []
    models = []
    all_preds = []
    for experiment_iter in range(n_iter):
        for i in range(10):
            normal_class = i
            
            #binary_y
            bin_y = all_y != normal_class; classes = [True, False]
            
            train_ind = get_subsample_indices(classes, bin_y, select=[n_train_anomaly, n_train_normal], random_state=random_state)
            test_ind = get_subsample_indices(classes, bin_y, select=[n_test_anomaly, n_test_normal], random_state=random_state)

            X_train = all_X[train_ind]
            X_test = all_X[test_ind]
            y_train = bin_y[train_ind]
            y_test = bin_y[test_ind]
        
            train_data.append((X_train, y_train))
            test_data.append((X_test, y_test))
        
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            perf_scores.append(metric(y_test, preds))
            
            models.append(model)
            all_preds.append(preds)
        
    return (np.mean(perf_scores), train_data, test_data, models, all_preds)
            
def benchmark_digits2(model, metric, n_iter=1, num_train_samples=1000, num_test_samples=1000, anomaly_ratio=0.05, random_state=None):
    
    random_state = check_random_state(random_state)
    
    n_train_normal = int(num_train_samples * (1.0 - anomaly_ratio))
    n_train_anomaly = num_train_samples - n_train_normal
    
    n_test_normal = int(num_test_samples * (1.0 - anomaly_ratio))
    n_test_anomaly = num_test_samples - n_test_normal
    
    all_X, all_y = load_digits_kaggle()
    
    perf_scores = []
    for experiment_iter in range(n_iter):
        for i in range(10):
            anomaly_class = i
            
            #binary y
            bin_y = all_y == anomaly_class; classes = [True, False]
            
            train_ind = get_subsample_indices(classes, bin_y, select=[n_train_anomaly, n_train_normal], random_state=random_state)
            test_ind = get_subsample_indices(classes, bin_y, select=[n_test_anomaly, n_test_normal], random_state=random_state)

            X_train = all_X[train_ind]
            X_test = all_X[test_ind]
            y_train = bin_y[train_ind]
            y_test = bin_y[test_ind]
        
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            perf_scores.append(metric(y_test, preds))
            
    return np.mean(perf_scores)
    
def show_digits(X, y, subplot=False, ax=None):
    if ax == None:
        fig = plt.figure()
    
    n = X.shape[0]
    
    dummy = np.zeros((28, 28))
    
    if subplot:
        for i in range(n):
            ax = fig.add_subplot(10, n/10, i+1)
            ax.imshow(X[i,:].reshape(28,28), cmap=plt.cm.gray)
    else:
        col = []
        count = 0
        for i in range(10):
            row = []
            for j in range(n/10):
                #dummy = X[count, :].reshape(28,28)
                dummy = np.empty((28, 28, 3), dtype='uint8')  # must use uint8 for imshow!!
                dummy[:, :, 0] = X[count, :].reshape(28, 28)
                if y[count]:
                    dummy[:, :, 1] = 0
                    dummy[:, :, 2] = 0
                else:
                    dummy[:, :, 1] = dummy[:, :, 0]
                    dummy[:, :, 2] = dummy[:, :, 0]
                #import pdb;pdb.set_trace()
                row.append(dummy)
                count += 1
            row = np.hstack(row)
            col.append(row)
        pixel_mat = np.vstack(col)
        ax.imshow(pixel_mat, aspect='auto')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
        
    

def show_digit_rankings(results):
    
    fig1 = plt.figure()
    
    for i in range(5):
        ax = fig1.add_subplot(5, 1, i+1)
        #fig = plt.figure(); ax = plt.gca()
        test_X, test_y = results[2][i]
        rankings = results[4][i]
        sorted_rankings_ind = np.argsort(rankings)[::-1]
        test_X = test_X[sorted_rankings_ind]
        test_y = test_y[sorted_rankings_ind]
        show_digits(test_X, test_y, ax=ax)
    fig1.subplots_adjust(left  = .001,
                        right = .999,
                        bottom = .001,
                        top = .999,
                        wspace = .001,
                        hspace = .001)
    fig2 = plt.figure()
    for i in range(5, 10):
        ax = fig2.add_subplot(5, 1, i - 4)
        #fig = plt.figure(); ax = plt.gca()
        test_X, test_y = results[2][i]
        rankings = results[4][i]
        sorted_rankings_ind = np.argsort(rankings)[::-1]
        test_X = test_X[sorted_rankings_ind]
        test_y = test_y[sorted_rankings_ind]
        show_digits(test_X, test_y, ax=ax)
    fig2.subplots_adjust(left  = .001,
                        right = .999,
                        bottom = .001,
                        top = .999,
                        wspace = .001,
                        hspace = .001)
    fig1.tight_layout()
    fig2.tight_layout()
    
    return [fig1, fig2]
  
def show_digits_pil(X, y_true, y_score=None, n_samples_per_row=50):
    '''
    Input:
        `X` - the (n by d) data matrix where n is the number of samples
        `y_true` - the true binary labels
        `y_score` - the scores used to rank the samples of X, optional
            if not given, will assume that y_true is already sorted in order from highest ranked to lowest
    '''
    if y_score is not None:
        sort_ind = np.argsort(y_score)[::-1]
        y_true = y_true[sort_ind]
        X = X[sort_ind]
    
    n_samples = X.shape[0]
    n_samples_per_row = min(n_samples_per_row, n_samples)
    img_width = int(n_samples_per_row * 28)
    img_height = int(np.ceil(float(n_samples) / n_samples_per_row) * 28)
    whole_img = Image.new('RGB', (img_width, img_height))
    
    for sample_i in range(n_samples):
        pixel_values = X[sample_i, :].reshape(28, 28)
        
        # Create img - red if anomaly, black if normal
        img = np.empty((28,28,3), 'uint8')
        
        img[:, :, 0] = pixel_values
        if y_true[sample_i]:
            img[:, :, 1] = 0
            img[:, :, 2] = 0
        else:
            img[:, :, 1] = pixel_values
            img[:, :, 2] = pixel_values
        
        img = Image.fromarray(img)
        
        # Paste into final image
        x_coord = sample_i % n_samples_per_row * 28
        y_coord = sample_i / n_samples_per_row * 28
        
        whole_img.paste(img, (x_coord, y_coord))
    
    return whole_img

    
# cached
#_digits_X, _digits_y = load_digits_kaggle()

if __name__ == '__main__':
    _create_digits_kaggle_memmap()
