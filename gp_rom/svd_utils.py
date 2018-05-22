__doc__ = """Functions to perform the SVD on a matrix,
    project feature vectors in that matrix into the SVD
    basis, and also deproject back to the original basis.
    Other SVD-related functions are here as well"""
__author__ = """Zoheyr Doctor, zoheyr@gmailcom"""

import numpy as np
from matplotlib import pyplot as plt

def svd_projection(matrix, center=True,get_errors=False, error_level=1e-5,error_type='constant'):
    """take the singular-value decomposition of a data matrix A 
    with feature vectors a.
    return the svd matrices as well as the data projected into the
    svd basis:
        M = UsV*
        c = V*m
    Covariances in matrix E can also be propagated to variances F in data 
    in SVD basis:
        F = V*EV

    Parameters
    ----------
    A : array-like (ndata,nfeatures).  Take the svd of this matrix.  
    get_errors: bool. whether to propagate errors through the svd
    error_level: float.  Error on input data   
    error_type: string.  'constant' or 'relative'.  Constant or relative
        error on input data is propagated to svd basis
    center: remove the mean of each feature in the data and return 
        the mean values

    Returns
    -------
    c: data matrix in the SVD basis
    U: left singular matrix
    s: matrix of singular values
    V: right singular matrix
    c_variance: variance on data in SVD basis
    center_values: mean values of the data matrix before SVD 
    """

    if center:
        feature_means = np.mean(A,axis=1)
        feature_variances = np.std(A,axis=1)
    U, s, VT = np.linalg.svd(A, full_matrices=True)
    V = VT.T
    c = np.array(map(lambda a: np.dot(a, V), A))
    
    if get_errors == True:
        if error_type=='relative':
            # constant relative error
            c_variance = np.array(map(lambda a: np.diag(np.dot(V.T,np.dot(np.diag(np.power(a,2.)),V))),error_level*A))
        else:
        # constant error 
            c_variance = np.array(map(lambda a: np.diag(np.dot(V.T,np.dot(np.diag(np.power(a,2.)),V))),error_level*np.ones_like(A)))
        return c,U,s,V,c_variance
    else:
        return c,U,s,V

def get_num_comps(s,variance_limit=0.9):
    """ get number of SVD components needed to capture
    variance_limit of the variance in the data

    Parameters
    ----------
    s: singular value matrix
    variance_limit: float, amount of variance to be captured

    Returns
    -------
    num_comps: number of components to get capture variance_limit
        of the variance in the data.
    """

    return np.amin(np.argwhere((np.cumsum(s)/np.sum(s)) > variance_limit))

def plot_svd_eigenvalues(s):
    """ Plot the svd eigenvalues of svd matrix s

    Parameters
    ----------
    s: singular value matrix

    Returns
    -------
    f: matplotlib figure
    ax: matplotlib axes
    """
    f,ax = plt.subplots(2,sharex=True)
    ax[0].plot(np.diag(s))
    ax[1].plot(np.diag(s)/np.amax(np.diag(s)))
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Eigenvalue Number')
    ax[0].set_ylabel('Normalized Eigenvalues')
    ax[1].set_ylabel('Cumulative Eigenvalues')

    return f,ax

def svd_deproject(c,V,errors=None):
    """
    Project data matrix c in SVD basis back to 
    original basis:
        a = Vc 

    Parameters
    ----------
    c: data matrix in SVD basis
    V: right singular matrix
    errors: 1-sigma errors on coefficients

    Returns
    -------
    a: data matrix in original basis
    a_variance: marginal variances on data
    """
    a = np.array(map(lambda A: np.dot(A, V.T), a)) 
    if not errors==None:
        a_variance = np.array(map(lambda A: np.diag(np.dot(V.T,np.dot(np.diag(np.power(A,2.)),V))),errors))
        return a,a_variance
    else:
        return a


