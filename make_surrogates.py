# -*- coding: utf-8 -*-
"""
Surrogate data methods.

Main Author: Caroline Cannistra (carolc24@uw.edu)
Contributing Author: Alex Yuan

References:
    
    [1]  Dimitris N. Politis & Joseph P. Romano (1994) The Stationary Bootstrap, Journal of the American Statistical Association, 89:428, 1303-1313, DOI: 10.1080/01621459.1994.10476870
    [2] Donges, Jonathan F., et al. "Unified functional network and nonlinear time series analysis for complex systems science: The pyunicorn package." Chaos: An Interdisciplinary Journal of Nonlinear Science 25.11 (2015).
    [3] Thiel, Marco & Romano, Maria & Kurths, Juergen & Rolfs, Martin & Kliegl, Reinhold. (2006). Twin surrogates to test for complex synchronisation. http://dx.doi.org/10.1209/epl/i2006-10147-0. 75. 10.1209/epl/i2006-10147-0. 
    [4] Lancaster, Gemma, et al. "Surrogate data for hypothesis testing of physical systems." Physics Reports 748 (2018): 1-60.
    [5] Yuan, Alex & Shou, Wenying. (2022). An exactly valid and distribution-free statistical significance test for correlations between time series. 10.1101/2022.01.25.477698. 

"""

import numpy as np
from pyunicorn.timeseries import surrogates
from scipy.spatial import distance_matrix
from ccm import setup_problem, choose_embed_params
from dependence_test_suite import delay_scan

def get_perm_surrogates(timeseries, n_surr=99):
    """
    Random shuffle method.

    Parameters
    ----------
    timeseries : numpy.array
        1D timeseries of length N.
    n_surr : int, optional
        Number of surrogates. The default is 99.

    Returns
    -------
    result : numpy.array
        2D array of surrogates with shape (n_surr,N).

    """
    sz = timeseries.size;
    result = np.zeros([sz,n_surr]);
    for col in range(n_surr):
        #shuffle points without replacement
        result[:,col] = np.random.choice(timeseries,size=sz,replace=False);
    return result;

#stationary bootstrap method from [1]
def get_stationary_bstrap_surrogates(timeseries, p_jump=0.05, n_surr=99):
    """
    
    Parameters
    ----------
    timeseries : numpy.array
        1D timeseries of length N.
    p_jump : float, optional
        Probability of jumping to a new position. The default is 0.05.
    n_surr : int, optional
        Number of surrogates. The default is 99.

    Returns
    -------
    result : numpy.array
        2D array of surrogates with shape (n_surr,N).

    """
    sz = timeseries.size
    result = np.zeros([sz, n_surr])
    result[0,:] = np.random.choice(sz, size=n_surr, replace=True)
    for col in range(n_surr):
    	for row in range(1,sz):
    		if np.random.random() < p_jump:
    			result[row,col] = np.random.choice(sz)
    		else:
    			result[row,col] = (result[row-1,col] + 1) % sz
    for col in range(n_surr):
    	for row in range(sz):
    		result[row,col] = timeseries[int(result[row,col])]
    return result

#iterated amplitude-adjusted fourier transform method
#from [2]
def get_iaaft_surr(data, n_surr=99, n_iter=200):
    """

    Parameters
    ----------
    data (numpy.array): 1D time series vector with length N
    n_surr (int): number of surrogates. The default is 99.
    n_iter (int): number of iterations to match power spectrum. 
        The default is 200.

    Returns
    -------
    (numpy.array): array of surrogates with shape (n_surr,N)

    """
    results = []
    i = 0
    while i < n_surr:
    	obj = surrogates.Surrogates(original_data=data.reshape(1,-1), silence_level=2)
    	surr = obj.refined_AAFT_surrogates(original_data=data.reshape(1,-1), n_iterations=n_iter, output='true_spectrum')
    	surr = surr.ravel()
    	if not np.isnan(np.sum(surr)):
    		results.append(surr)
    		i += 1
    return np.array(results).T

#helper function for twin method
def get_maxnorm_distmat(X, Y):
    """
    returns max norm distance matrix

    Args:
        X (array): an m-by-n array where m is the number of vectors and n is the
            vector length
        Y (array): same shape as X
    """
    n_vecs, n_dims = X.shape
    K_by_dim = np.zeros([n_dims, n_vecs, n_vecs])
    for dim in range(n_dims):
        K_by_dim[dim,:,:] = distance_matrix(X[:,dim].reshape(-1,1), Y[:,dim].reshape(-1,1))
    return K_by_dim.max(axis=0)

#helper function for twin method
def choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency=0.12, distmat_fxn=get_maxnorm_distmat):
    """Given a univariate timeseries, embedding parameters, and a twin frequency,
		choose the twin threshold.

	   Args:
	     timeseries (numpy array): a univariate time series
		 embed_dim (int): embedding dimension
		 tau (int): embedding delay
		 neighbor_frequency (float): Fraction of the "recurrence plot" to choose
		 	as neighbors. Note that not all neighbors are twins.
        distmat_fxn (function): function that returns a distance matrix.
            default is get_maxnorm_distmat.

		Returns:
		  recurrence distance threshold for twins
    """
    # timeseries is 1d
    timeseries = np.copy(timeseries.flatten())
    data_ = np.zeros([timeseries.size, 2])
    data_[:,0] = timeseries
    data_[:,1] = timeseries
    X, y = setup_problem(data_, embed_dim=embed_dim, tau=tau)
    K = distmat_fxn(X,X)
    #np.fill_diagonal(K, np.inf) # self-neighbors are allowed in recurrence plot.
    k = K.flatten()
    k = np.sort(k)
    idx = np.floor(k.size * neighbor_frequency).astype(np.int)
    return k[idx]

#twin method
#from [2], described in [3]
def get_twin_surrogates(timeseries, embed_dim=None, tau=None, 
                        num_surr=99, neighbor_frequency=0.1, th=None):
    """Given a univariate time series, generate num_surr twin surrogates.

	   Args:
	     timeseries (numpy array): a univariate time series
		 embed_dim (int): embedding dimension
		 tau (int): embedding delay
		 neighbor_frequency (float): Fraction of the "recurrence plot" to choose
		 	as neighbors. Note that not all neighbors are twins.
        th (float): distance threshold for choosing twins. if None,
            the method will choose th based on neighbor frequency.

		Returns:
		  recurrence distance threshold for twins
    """
    if embed_dim is None or tau is None:
        embed_dim, tau = choose_embed_params(timeseries)
    if th is None:
    	th = choose_twin_threshold(timeseries, embed_dim, tau, neighbor_frequency)
    results = [] # i=0
    obj = surrogates.Surrogates(original_data=timeseries.reshape(1,-1), silence_level=2)
    for i in range(num_surr):
    	surr = obj.twin_surrogates(original_data=timeseries.reshape(1,-1), dimension=embed_dim, delay=tau, threshold=th, min_dist=1)
    	surr = surr.ravel()
    	results.append(surr)
    return np.array(results).T

#circular permutation method
def get_circperm_surrogates(timeseries):
    """given a univariate time series of length N,
    makes N circular permutation surrogates.
    
    Parameters
    ----------
    timeseries (numpy.array): 1D time series with shape (N,)

    Returns
    -------
    result (numpy.array): 2D time series with shape (N,N)

    """
    result = np.zeros([timeseries.size, timeseries.size])
    for i in range(timeseries.size):
    	result[:,i] = np.roll(timeseries, i)
    return result

#truncated time shift
#described in [5]
def tts(x,y,r,statistic,maxlag=0):
    """
    Perform a test for independence between time series x and y
    using a given statistic and the truncated time shift 
    surrogate data method. If using lags, pick the best lag
    for original data AND surrogate data.

    Parameters
    ----------
    x (numpy.array): 1D time series of length N
    y (numpy.array): 1D time series of length N
    r (int): truncation radius.
    statistic (function): correlation statistic
    maxlag (int): maximum time lag

    Returns
    -------
    pval (float): probability of getting this or a more extreme result
    under the null hypothesis.

    """
    # time shift
    t = len(x); #number of time points in orig data
    xstar = x[r:(t-r)]; # middle chunk of data
    tstar = len(xstar); #length of middle chunk of data
    ystar = y[r:(t-r)];
    t0 = r;
    
    importance_true = np.max(delay_scan(xstar, ystar.reshape(-1,1),statistic,maxlag)[1]);
    
    y_surr = np.tile(ystar,[2*r+1, 1])
    
    #iterate through all shifts from -r to r
    for shift in np.arange(t0 - r, t0 + r+1):
        
        y_surr[shift-t0] = y[shift:(shift+tstar)];
         
    #pick highest score for each surrogate
    importance_surr = np.max(delay_scan(xstar, y_surr.T, statistic,maxlag)[1:],axis=1);
    sig = np.mean(importance_surr >= importance_true, axis=0);
    return sig;

#truncated time shift
def tts_bad(x,y,r,statistic,maxlag=0):
    """
    Perform a test for independence between time series x and y
    using a given statistic and the truncated time shift 
    surrogate data method. If using lags, pick the best lag
    for original data and use the SAME lag for surrogate data.

    Parameters
    ----------
    x (numpy.array): 1D time series of length N
    y (numpy.array): 1D time series of length N
    r (int): truncation radius.
    statistic (function): correlation statistic
    maxlag (int): maximum time lag

    Returns
    -------
    pval (float): probability of getting this or a more extreme result
    under the null hypothesis.

    """
    # time shift
    t = len(x); #number of time points in orig data
    xstar = x[r:(t-r)]; # middle chunk of data
    tstar = len(xstar); #length of middle chunk of data
    ystar = y[r:(t-r)];
    t0 = r;
    
    score_list = delay_scan(xstar, ystar.reshape(-1,1),statistic,maxlag);
    [lag,importance_true] = score_list[:,np.argmax(score_list[1])]
    lag = int(lag)
    
    y_surr = np.tile(ystar,[2*r+1, 1])
    
    #iterate through all shifts from -r to r
    for shift in np.arange(t0 - r, t0 + r+1):
        
        y_surr[shift-t0] = y[shift:(shift+tstar)];
         
    #calculate surrogate scores based on original best lag
    if lag == 0:
        importance_surr = statistic(xstar, y_surr.T)
    elif lag > 0:
        importance_surr = statistic(xstar[lag:],(y_surr.T)[:-lag])
    else:
        importance_surr = statistic(xstar[:lag],(y_surr.T)[-lag:])
    sig = np.mean(importance_surr >= importance_true, axis=0);
    return sig;