""" gaussianfield.py

Python implementation of the Gaussian Field harmonic classifier from Zhu 2003 (ICML)
"""

import numpy as np

def combinatorial_laplacian(W):
    """ compute the combinatorial laplacian given edge weight matrix W """
    # the degree matrix -- sum up edge weights for each node
    D = np.diag(np.sum(W, axis=1))

    # the combinatorial laplacian
    laplacian = D - W
    return laplacian


def solve(W, labels, observed):
    """  Compute solution to the harmonic function from Zhu 2003 (ICML)

    Careful: the workshop paper has an error in equation 3
    Follow equation 5 from the ICML conference paper instead

    Args:
        W: (n, n) matrix of edge weights, W has to be symmetric, and all entries has to be non-negative
        labels: (n_observed, n_classes) class label array (one-hot encoding)
        observed: n-element indicator vector for the observed datapoints

    Returns:
        field: Gaussian Field values at unobserved (unlabeled) points
        laplacian_uu_inv: inverted Laplacian submatrix for unobserved (unlabeled) points
    """

    laplacian = combinatorial_laplacian(W)

    # invDeltaU = full(inv(Delta(U,U)))

    # partition the laplacian into labeled and unlabeled blocks...
    laplacian_uu = laplacian[np.ix_(~observed, ~observed)]
    W_ul = W[np.ix_(~observed, observed)]

    # Naive solution to the gaussian field
    # TODO: implement the efficient inverse laplacian update
    laplacian_uu_inv = np.linalg.inv(laplacian_uu)
    field = laplacian_uu_inv.dot(W_ul).dot(labels)

    return field, laplacian_uu_inv

def expected_risk(field, Linv):
    """ Compute the expected risk of the classifier f+(k), i.e. after adding each potential query k

    Args:
        field: Gaussian Field values at unobserved (unlabeled) points
        Linv: inverted Laplacian matrix for unobserved (unlabeled) points

    Returns:
        risk: vector with one entry for each potential query
    """

    # translate Zhu's vectorized matlab code from active_learning.m here...

    n_unobserved, nclasses = field.shape

    # divide kth column-wise by diagonal elements
    nG = Linv / np.diag(Linv)[:,np.newaxis].T

    # for each possible label
    risk = np.zeros(n_unobserved)

    # compute prefactor:
    pre_maxfplus = np.zeros((n_unobserved,n_unobserved))
    for c in range(nclasses):
        np.maximum(
            field[:,c][:,None] - field[:,c].T * nG,
            pre_maxfplus,
            out=pre_maxfplus
        )

    # nG = invDeltaU ./ repmat(diag(invDeltaU)', u, 1);
    for c in range(nclasses):
        # risk += sum(
        #     1- max(repmat(f(:,c), 1, u) + repmat(1-f(:,c).T, u, 1).*(invDeltaU ./ repmat(diag(invDeltaU).T, u, 1))
        #            , pre_maxfplus)
        # ) .* f(:,c).T;

        risk += np.sum(1 - np.maximum(field[:,c][:,None] + (1-field[:,c].T) * nG, pre_maxfplus), axis=0) * field[:,c].T

    return risk
