""" Naive implementation of Zhu paper
Zhu starts with just one labeled instance per class
they set the scale parameter to 380, but also thin the graph to 10 nearest neighbors.
"""

import numpy as np
from skimage import filters

def combinatorial_laplacian(W):
    """ compute the combinatorial laplacian given edge weight matrix W """
    # the degree matrix -- sum up edge weights for each node
    D = np.diag(np.sum(W, axis=1))

    # the combinatorial laplacian
    L = D - W
    return L


def solve_gaussian_field(W, labels, unlabeled):
    """  Compute solution to the harmonic function from Zhu 2003 (ICML) 

    Args:
        W: nxn matrix of edge weights
        labels: class label array (dense encoding)
        unlabeled: n-element indicator vector for the unlabeled datapoints

    Returns:
        field: Gaussian Field values at unlabeled points
        L_uu_inv: inverted Laplacian submatrix for unlabeled points
    """
    L = combinatorial_laplacian(W)

    # partition the laplacian into labeled and unlabeled blocks...
    L_uu = L[np.ix_(unlabeled, unlabeled)]
    L_ul = L[np.ix_(unlabeled, ~unlabeled)]
    
    # Naive solution to the gaussian field
    L_uu_inv = np.linalg.inv(L_uu)
    field = -L_uu_inv.dot(L_ul).dot(labels[~unlabeled])
    
    return field, L_uu_inv

def estimated_risk(field):
    """ Estimate the risk for the output of Gaussian Field model 

    Args:
        field: vector containing Gaussian Field values

    Returns:
        risk: vector containing estimated risk
    """
    # evaluate the classifier (threshold at 0.5 for the Bayes classifier)
    sgn = field > filters.threshold_otsu(field)
    risk = np.sum(1-field[sgn != 0]) + np.sum(field[sgn != 1])
    return risk
    

def expected_risk(field, L_uu_inv):
    """ Compute the expected risk of the classifier f+(k), i.e. after adding each potential query k

    Args:
        field: Gaussian Field values at unlabeled points
        L_uu_inv: inverted Laplacian matrix for unlabeled points
        unlabeled: indicator
    return a `risk` vector with one entry for each potential query
    """
    fn = [
        lambda f: 1 - f,
        lambda f: f
    ]

    # Consider the estimated risk after adding each potential query point k
    risk = []
    n_unlabeled = L_uu_inv.shape[0]
    
    for k in range(n_unlabeled):
        risk.append(0)

        # consider each possible value y_k could take:
        for y_k in (0, 1):
            f_est = field + (y_k - field[k]) * L_uu_inv[k] / L_uu_inv[k,k]
            risk[k] += fn[y_k](field[k]) * estimated_risk(f_est)
    
    return risk
