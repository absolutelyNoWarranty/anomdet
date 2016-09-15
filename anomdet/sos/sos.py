# Original Stochastic Outlier Selection functions

#!/usr/bin/env python
#
# Stochastic Outlier Selection
# 
# Copyright (c) 2013, Jeroen Janssens
# All rights reserved.
#
# Distributed under the terms of the BSD Simplified License.
# The full license is in the LICENSE file, distributed with this software.
#
# For more information about SOS, see https://github.com/jeroenjanssens/sos
# J.H.M. Janssens, F. Huszar, E.O. Postma, and H.J. van den Herik. Stochastic
# Outlier Selection. Technical Report TiCC TR 2012-001, Tilburg University,
# Tilburg, the Netherlands, 2012.
#
# Please note that because SOS is inspired by t-SNE (created by Laurens 
# van der Maaten; see http://homepage.tudelft.nl/q19j49/t-SNE.html),
# this code borrows functionality from the Python implementation,
# namely the functions x2p and Hbeta.


import argparse
import logging
import numpy as np
import sys


#log_format = '%(asctime)-15s  [%(levelname)s] - %(name)s: %(message)s'
#logging.basicConfig(format=log_format, level=logging.INFO)
#log = logging.getLogger('SOS')


def x2d(X, metric, logger=None):
    """Computer dissimilarity matrix."""

    metric = metric.lower()
    (n, d) = X.shape
    logger.debug("The data set is %dx%d", n, d)
    if metric == 'none':
        if n != d:
            logger.error(("If you specify 'none' as the metric, the data set "
                "should be a square dissimilarity matrix"))
            exit(1)
        else:
            logger.debug("The data set is a dissimilarity matrix")
            D = X
    #elif metric == 'euclidean':
        #logger.debug("Computing dissimilarity matrix using Euclidean metric")
        #sumX = np.sum(np.square(X), 1)
        #D = np.sqrt(np.add(np.add(-2 * np.dot(X, X.T), sumX).T, sumX))
        # NUMERICALLY UNSTABLE!
    else:
        try:
            from scipy.spatial import distance
        except ImportError as e:
            logger.error(("Please install scipy if you wish to use a metric "
                "other than 'euclidean' or 'none'"))
            exit(1)
        else:
            logger.debug("Computing dissimilarity matrix using %s metric",
                metric.capitalize())
            D = distance.squareform(distance.pdist(X, metric))
    return D

def init_nonzero(n):
    '''
    Generate numbers close to zero but not zero
    '''
    res = np.random.randn(n)
    mask = res >= 0
    res[mask] = res[mask] + 1e-5
    res[np.logical_not(mask)] = res[mask] - 1e-5
    return res

def d2a(D, perplexity, tol=1e-5, logger=None):
    """Return affinity matrix.

    Performs a binary search to get affinities in such a way that each
    conditional Gaussian has the same perplexity.

    """

    (n, _) = D.shape
    A = np.zeros((n, n))
    beta = 10.0 + np.abs(np.random.randn(n))
    ##beta = init_nonzero(n) 
    #beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        if i % 100 == 0:
            logger.debug("Computing affinities (%d/%d)", i, n)

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = 0.0 ##betamin = -2.0  ##
        betamax = 20.0 ##betamax = 2.0   ##
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisA) = get_perplexity(Di, beta[i], logger=logger)

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            #import pdb; pdb.set_trace()
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.0
            else:
                betamax = beta[i]
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.0
            # Recompute the values
            (H, thisA) = get_perplexity(Di, beta[i], logger=logger)
            Hdiff = H - logU
            tries += 1
            logger.debug("Hdiff : %f", Hdiff)
        # Set the final row of A
        A[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisA

    logger.debug("Computing affinities (%d/%d)", n, n)
    logger.debug("Hdiff : %f", Hdiff)
    return A


def get_perplexity(D, beta, logger=None):
    """Compute the perplexity and the A-row for a specific value of the
    precision of a Gaussian distribution.

    """

    A = np.exp(-D * beta)
    sumA = sum(A)
    H = np.log(sumA) + beta * np.sum(D * A) / sumA
    return H, A


def a2b(A, logger=None):
    logger.debug("Computing binding probabilities")
    B = A / A.sum(axis=1)[:,np.newaxis]
    return B


def b2o(B, logger=None):
    logger.debug("Computing outlier probabilities")
    O = np.prod(1-B, 0)
    return O


def sos(X, metric, perplexity, logger=None):
    D = x2d(X, metric, logger=logger)
    
    # Normalize distance matrix so that distances aren't too large
    # (for numerical reasons)
    D = D / np.mean(D)
    
    A = d2a(D, perplexity, logger=logger)
    B = a2b(A, logger=logger)
    #if args.binding_matrix:
    #    np.savetxt(args.output, B, '%1.8f', delimiter=',')
    #    exit()
    O = b2o(B, logger=logger)
    return O