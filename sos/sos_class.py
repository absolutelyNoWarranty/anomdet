# Wrapper for Stochastic Outlier Selection

from ..base import BaseAnomalyDetector
from .sos import sos
import logging

from sklearn.preprocessing import StandardScaler

class StochasticOutlierSelection(BaseAnomalyDetector):
    """Calculate Stochastic Outlier Selection

    Parameters
    ----------

    metric : string
        Metric used to compute the dissimilarity matrix.
        default - 'euclidean'
        
    perplexity : float

    save_binding_matrix_to : str, where to save the binding matrix to if
        given. Default : None
    
    verbose : bool, default : False
    
    Returns
    -------

    Outlier probabilities

    References
    ----------
    Technical report: J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. Stochastic Outlier Selection. Technical Report TiCC TR 2012-001, Tilburg University, Tilburg, the Netherlands, 2012.
    """
    
    def __init__(self, metric='euclidean', perplexity=30.0, verbose=False, standard_scale=False):
        self.metric = metric
        self.perplexity = perplexity
        self.verbose = verbose
        # Warning: for some reason sos by itself doesn't work on MNIST (and other datasets I assume)
        # unless you standarize the dataset first. It's really weird.
        self.standard_scale=standard_scale
    
    def predict(self, X=None):
        """Calculate connectivity-based outlier factor for each sample in A

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        probs : array, shape (n_samples,)
            Outlier probabilities determined by stochastic outlier selection.
        """
        
        if X is None:
            if self.X_ is None:
                raise Exception("No data")
            X = self.X_
        
        log_format = '%(asctime)-15s  [%(levelname)s] - %(name)s: %(message)s'
        logging.basicConfig(format=log_format, level=logging.INFO)
        logger = logging.getLogger('SOS')
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.ERROR)
        
        if self.standard_scale:
            X = StandardScaler().fit_transform(X.copy())
        return sos(X, self.metric, self.perplexity, logger=logger)
    
    def fit(self, X=None, y=None):
        self.X_ = X
        return self