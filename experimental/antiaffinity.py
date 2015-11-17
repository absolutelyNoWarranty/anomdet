from ..base import BaseAnomalyDetector

import numpy as np
from sklearn.utils import as_float_array
from sklearn.metrics import euclidean_distances

# sklearn's affinity_propagation, modified to return A and R matrices
def affinity_propagation(S, preference=None, convergence_iter=15, max_iter=200,
                         damping=0.5, copy=True, verbose=False):
    """Perform Affinity Propagation Clustering of data

    Parameters
    ----------

    S : array-like, shape (n_samples, n_samples)
        Matrix of similarities between points

    preference : array-like, shape (n_samples,) or float, optional
        Preferences for each point - points with larger values of
        preferences are more likely to be chosen as exemplars. The number of
        exemplars, i.e. of clusters, is influenced by the input preferences
        value. If the preferences are not passed as arguments, they will be
        set to the median of the input similarities (resulting in a moderate
        number of clusters). For a smaller amount of clusters, this can be set
        to the minimum value of the similarities.

    convergence_iter : int, optional, default: 15
        Number of iterations with no change in the number
        of estimated clusters that stops the convergence.

    max_iter : int, optional, default: 200
        Maximum number of iterations

    damping : float, optional, default: 0.5
        Damping factor between 0.5 and 1.

    copy : boolean, optional, default: True
        If copy is False, the affinity matrix is modified inplace by the
        algorithm, for memory efficiency

    verbose : boolean, optional, default: False
        The verbosity level

    Returns
    -------

    cluster_centers_indices : array, shape (n_clusters,)
        index of clusters centers

    labels : array, shape (n_samples,)
        cluster labels for each point

    Notes
    -----
    See examples/cluster/plot_affinity_propagation.py for an example.

    References
    ----------
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007
    """
    S = as_float_array(S, copy=copy)
    n_samples = S.shape[0]

    if S.shape[0] != S.shape[1]:
        raise ValueError("S must be a square array (shape=%s)" % repr(S.shape))

    if preference is None:
        preference = np.median(S)
    if damping < 0.5 or damping >= 1:
        raise ValueError('damping must be >= 0.5 and < 1')

    random_state = np.random.RandomState(0)

    # Place preference on the diagonal of S
    S.flat[::(n_samples + 1)] = preference

    A = np.zeros((n_samples, n_samples))
    R = np.zeros((n_samples, n_samples))  # Initialize messages

    # Remove degeneracies
    S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) *
          random_state.randn(n_samples, n_samples))

    # Execute parallel affinity propagation updates
    e = np.zeros((n_samples, convergence_iter))

    ind = np.arange(n_samples)

    for it in range(max_iter):
        # Compute responsibilities
        Rold = R.copy()
        AS = A + S

        I = np.argmax(AS, axis=1)
        Y = AS[np.arange(n_samples), I]  # np.max(AS, axis=1)

        AS[ind, I[ind]] = - np.finfo(np.double).max

        Y2 = np.max(AS, axis=1)
        R = S - Y[:, np.newaxis]

        R[ind, I[ind]] = S[ind, I[ind]] - Y2[ind]

        R = (1 - damping) * R + damping * Rold  # Damping

        # Compute availabilities
        Aold = A
        Rp = np.maximum(R, 0)
        Rp.flat[::n_samples + 1] = R.flat[::n_samples + 1]

        A = np.sum(Rp, axis=0)[np.newaxis, :] - Rp

        dA = np.diag(A)
        A = np.minimum(A, 0)

        A.flat[::n_samples + 1] = dA

        A = (1 - damping) * A + damping * Aold  # Damping

        # Check for convergence
        E = (np.diag(A) + np.diag(R)) > 0
        e[:, it % convergence_iter] = E
        K = np.sum(E, axis=0)

        if it >= convergence_iter:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se == convergence_iter) + (se == 0))
                           != n_samples)
            if (not unconverged and (K > 0)) or (it == max_iter):
                if verbose:
                    print("Converged after %d iterations." % it)
                break
    else:
        if verbose:
            print("Did not converge")

    I = np.where(np.diag(A + R) > 0)[0]
    K = I.size  # Identify exemplars

    if K > 0:
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)  # Identify clusters
        # Refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.where(c == k)[0]
            j = np.argmax(np.sum(S[ii[:, np.newaxis], ii], axis=0))
            I[k] = ii[j]

        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(K)
        labels = I[c]
        # Reduce labels to a sorted, gapless, list
        cluster_centers_indices = np.unique(labels)
        labels = np.searchsorted(cluster_centers_indices, labels)
    else:
        labels = np.empty((n_samples, 1))
        cluster_centers_indices = None
        labels.fill(np.nan)

    return R, A	

class Dummy(BaseAnomalyDetector):
    def __init__(self, strategy, ref):
        self.strategy = strategy
        self.ref = ref
        
    def fit(self, X):
        self.ref.fit(X)
        return self
        
    def predict(self, X):
        return self.ref.predict(X, self.strategy)
class AntiAffinityMaker(BaseAnomalyDetector):
    """ Use Affinity Propagation Clustering to Detection anomalies
    """
    
    
    def __init__(self, damping=.5, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
    
    @property
    def just_A(self):
        return Dummy(strategy="just_A", ref=self)
    
    @property
    def just_R(self):
        return Dummy(strategy="just_R", ref=self)
    
    @property
    def both(self):
        return Dummy(strategy="both", ref=self)
    
    def fit(self, X):
        """ Create affinity matrix from negative euclidean distances, then
        apply affinity propagation clustering to get A and R matrices.

        Parameters
        ----------

        X: array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.
            
        Returns
        -------
        
        outlier_scores : diagonals of R minus diagonals of A
        
        """
        if not hasattr(self, 'A'):
            if self.affinity == "precomputed":
                self.affinity_matrix_ = X
            elif self.affinity == "euclidean":
                self.affinity_matrix_ = -euclidean_distances(X, squared=True)
            else:
                raise ValueError("Affinity must be 'precomputed' or "
                                 "'euclidean'. Got %s instead"
                                 % str(self.affinity))

            self.R, self.A = affinity_propagation(
                self.affinity_matrix_, self.preference, max_iter=self.max_iter,
                convergence_iter=self.convergence_iter, damping=self.damping,
                copy=self.copy, verbose=self.verbose)

        return self
    
    def predict(self, X, strategy):
        ### !!! ASSUMING PREDICT X AND FIT WERE THE SAME LOLOLOLLO
        if strategy == "just_R":
            return np.diag(self.R)
        elif strategy == "just_A":
            return -np.diag(self.A)
        elif strategy == "both":
            return np.diag(self.R) - np.diag(self.A)
            
            
class AntiAffinity(BaseAnomalyDetector):
    """ Use Affinity Propagation Clustering to Detection anomalies
    """
    
    
    def __init__(self, damping=.9, max_iter=200, convergence_iter=15,
                 copy=True, preference=None, affinity='euclidean',
                 predict_with="both", a=None, b=None, f=None, verbose=False):

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.copy = copy
        self.verbose = verbose
        self.preference = preference
        self.affinity = affinity
        self.predict_with = predict_with    
        self.a = a
        self.b = b
        self.f = f

    def fit(self, X, y=None):
        """ Create affinity matrix from negative euclidean distances, then
        apply affinity propagation clustering to get A and R matrices.

        Parameters
        ----------

        X: array-like, shape (n_samples, n_features) or (n_samples, n_samples)
            Data matrix or, if affinity is ``precomputed``, matrix of
            similarities / affinities.
            
        Returns
        -------
        
        outlier_scores : diagonals of R minus diagonals of A
        
        """

        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
        elif self.affinity == "euclidean":
            self.affinity_matrix_ = -euclidean_distances(X, squared=False)
        elif self.affinity == "euclidean_distance":
            self.affinity_matrix_ = euclidean_distances(X, squared=False)
        else:
            raise ValueError("Affinity must be 'precomputed' or "
                             "'euclidean'. Got %s instead"
                             % str(self.affinity))

        self.R, self.A = affinity_propagation(
            self.affinity_matrix_, self.preference, max_iter=self.max_iter,
            convergence_iter=self.convergence_iter, damping=self.damping,
            copy=self.copy, verbose=self.verbose)

        return self
    
    def predict(self, X):
        if self.f is not None:
            return self.f(self.R, self.A)
        if self.a is not None and self.b is not None:
            return self.a * np.diag(self.R) + self.b * np.diag(self.A)
    
        ### !!! ASSUMING PREDICT X AND FIT WERE THE SAME LOLOLOLLO
        if self.predict_with == "both":
            return np.diag(self.R) - np.diag(self.A)
        elif self.predict_with == "just_R":
            return np.diag(self.R)
        elif self.predict_with == "just_A":
            return -np.diag(self.A)
        elif isinstance(self.predict_with, float):
            return self.predict_with * np.diag(self.R) - (1.0 - self.predict_with) * np.diag(self.A)