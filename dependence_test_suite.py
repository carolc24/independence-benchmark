# -*- coding: utf-8 -*-
"""
Run dependence tests.

Main author: Caroline Cannistra (carolc24@uw.edu)
Contributing author: Alex Yuan

References: 
    [1] Lancaster, Gemma, et al. "Surrogate data for hypothesis testing of physical systems." Physics Reports 748 (2018): 1-60.
    [2] Yuan, Alex & Shou, Wenying. (2022). An exactly valid and distribution-free statistical significance test for correlations between time series. 10.1101/2022.01.25.477698. 

"""

import numpy as np
from statistics import *
from make_surrogates import *

#adds delays to any statistic
def delay_scan(x,y,statistic,maxlag=5,kw_statistic={}):
    """  
    Finds correlation statistic between two 
    timeseries at different time delays.
    
    Parameters
    ----------
    x : numpy.array
        1D time series of length N.
    y : numpy.array
        Array of M time series with shape (M,N).
    statistic : function
        Correlation function.
    maxlag : int, optional
        Maximum time lag between x and y. The default is 5.
    kw_statistic : dict, optional
        Arguments to pass to the statistic function. The default is {}.

    Returns
    -------
    result : numpy.array
        Has shape (2,2*maxlag+1).
        First row: time lag
        Second row: score

    """
    score_sim = statistic(x,y,**kw_statistic);
    # rows: # of y rows
    # columns: # of lags to test
    score = np.zeros((score_sim.size,2*maxlag+1));
    score[:,maxlag] = score_sim
    lags = np.arange(-2*maxlag,2*maxlag+1,2);
    if (maxlag > 0):
        for i in np.arange(1,maxlag+1):
            score[:,maxlag+i] = statistic(x[2*i:],y[:-2*i],**kw_statistic);
            score[:,maxlag-i] = statistic(x[:-2*i],y[2*i:],**kw_statistic);
    return np.vstack((lags,score));

def sig_test_good(x,y,statistic,surr_fxn,maxlag):
    """
    Perform a delayed correlation test between two time series
    by picking the time lag with the highest correlation
    for the actual time series and for the surrogate data.
    Corresponds to Part iv of Figure 3 and panel E of figure 4.

    Parameters
    ----------
    x : numpy.array
        1D time series vector of length N.
    y : numpy.array
        1D time series vector of length N, template for surrogates
    statistic : function
        Function that calculates correlation.
    surr_fxn : function
        Function that makes surrogates.
    maxlag : int
        Maximum time lag.

    Returns
    -------
    pval : float
        Probability of getting this or a higher score under
        the null hypothesis of independence.

    """
    #find statistic from original data
    score = np.max(delay_scan(x,y.reshape(-1,1),statistic,maxlag)[1]);
    #make surrogate data
    surr = surr_fxn(y);
    #truncate original x to match surrogate y if necessary
    x = x[:surr.shape[0]]
    #find null statistic for each surrogate 
    #using same maximizing procedure as original
    null = np.max(delay_scan(x,surr,statistic,maxlag)[1:],axis=1);
    #one tailed test
    pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
    return pval;

def sig_test_bad(x,y,statistic,surr_fxn,maxlag):
    """
    Perform a delayed correlation test between two time series
    by picking the time lag with the highest correlation
    for the actual time series and using the SAME time lag
    for surrogate data.
    Corresponds to Part iii of Figure 3 and panel D of figure 4.

    Parameters
    ----------
    x : numpy.array
        1D time series vector of length N.
    y : numpy.array
        1D time series vector of length N, template for surrogates
    statistic : function
        Function that calculates correlation.
    surr_fxn : function
        Function that makes surrogates.
    maxlag : int
        Maximum time lag.

    Returns
    -------
    pval : float
        Probability of getting this or a higher score under
        the null hypothesis of independence.

    """
    #find statistic from original data and lag
    score_list = delay_scan(x,y.reshape(-1,1),statistic,maxlag);
    [lag,score] = score_list[:,np.argmax(score_list[1])]
    lag = int(lag);
    #get surrogate data
    surr = surr_fxn(y);
    #truncate x if necessary
    x = x[:surr.shape[0]]
    #find correlation for shifted x and surrogates
    if lag == 0:    
        null = statistic(x,surr);
    elif lag > 0:
        null = statistic(x[lag:],surr[:-lag]);
    else:
        null = statistic(x[:lag],surr[-lag:]);
    #one tailed test
    pval = (np.sum(null >= score) + 1) / (np.size(null) + 1);
    return pval;    

#Algorithm from [1]
#Trim data so beginning matches end
def trim_periodic_data(y,p=0):
    """
    Trim data so the beginning matches the end.

    Parameters
    ----------
    y : numpy.array
        1D time series of length N.
    p : int, optional
        Length of time series segment to match.
        If 0, the function will pick a value based on
        the period of y.

    Returns
    -------
    k_start : int
        The starting index of the trimmed time series.
    k_end : int
        The ending index of the trimmed time series.

    """
    if (p == 0):
        #Find period of signal (very simple and messy method)
        
        #subtract mean to get rid of 0 peak
        y_clean = y - np.mean(y);
        #get fourier transform
        fft = np.fft.rfft(y_clean)
        #first peak of fft
        freq = np.argmax(np.abs(fft))
        T = len(y)/freq/2
        p = int(np.ceil(T/10))
    
    #Now, run trimming algorithm
    trunc_max = int(np.floor(len(y)/10));
    match_mat = np.zeros((trunc_max,trunc_max))
    
    #grid search for best k1 and k2
    for k1 in np.arange(trunc_max):
        for k2 in np.arange(len(y) - trunc_max - p + 1, len(y) - p + 1):
            y_left = y[k1:k1+p]
            y_right = y[k2:k2+p]
            match_mat[k1,len(y)-p-k2] = np.sum((y_left - y_right)**2);
    
    k_start, k_end = np.unravel_index(np.argmin(match_mat),match_mat.shape)
    #change from index to actual k2
    k_end = len(y) - p - k_end;
    
    return (k_start,k_end);

#from alex
def multishift_sigf(x, y, test_name, statistic, shift_grid, r_tts):
    """
    returns pvalues obtained from shifts and bonferroni correction.
    You can use this with any independence test, but we only use it
    with the truncated time shift test.

    Parameters
    ----------
    x : numpy.array
        1D time series of length N.
    y : numpy.array
        1D time series of length N.
    test_name : string
        Name of the surrogate test to use.
    statistic : function
        Correlation function.
    shift_grid : numpy.array
        1D array of integer time lags.
    r_tts : int
        Truncation radius if using truncated time shift.

    Returns
    -------
    pval : float
        Probability of seeing this or a more extreme result
        under the null hypothesis (zero dependence between
        x and y at any of the time delays in shift_grid).

    """
    if (2 * r_tts >= x.size) and (test_name == 'tts'):
        return np.nan
    pvals = np.zeros(shift_grid.size)
    for i, shift in enumerate(shift_grid):
        if shift >= 0:
            x_shifted = x[shift:]
            y_shifted = y[:y.size-shift]
        else:
            x_shifted = x[:x.size+shift]
            y_shifted = y[-shift:]
        pvals[i] = get_sigf(x_shifted, y_shifted, statistic,r_tts=r_tts, test_list=[test_name])[test_name]
    return np.min(pvals) * shift_grid.size # smallest pval multiplied by number of lags

def choose_r(N):
    """
    Pick truncation radius for truncated time shift test.

    Parameters
    ----------
    N : int
        Number of time points.

    Returns
    -------
    r : int
        Truncation radius close to 1/4 of N.
        r+1 is a multiple of 20.

    """
    delta = (N/4 + 1) % 20
    return int(N/4 - delta)

def r_multishift(N,maxlag,alpha=0.05):
    """
    Pick truncation radius for multishift version of 
    truncated time shift test.

    Parameters
    ----------
    N : int
        Number of time points.
    maxlag : int
        Maximum time lag.
    alpha : float, optional
        Desired type I error rate. The default is 0.05.

    Returns
    -------
    r : int
        Truncation radius.

    """
    m = 2*maxlag + 1;
    # truncated time series should be at least 80 time points or else granger gets mad
    r_options = np.arange(int(m/alpha) - 1, N/2 - 40, int(m/alpha)); 
    if r_options.size == 0:
        print("Time series too short for multishift with maxlag=%d" % (maxlag));
    return int(np.max(r_options));

#run suite of tests
def get_sigf(x, y, statistic, test_list='all', maxlag=0, kw_randphase={}, \
             kw_bstrap={}, kw_twin={}, r_tts=choose_r, r_naive=choose_r, 
             sig_test=sig_test_good, tts_test=tts):
    """
    Run a set of tests on two time series with a given statistic.

    Parameters
    ----------
    x : numpy.array
        1D time series of length N.
    y : TYPE
        1D time series of length N.
    statistic : function
        Correlation function.
    test_list : string or list, optional
        Either 'all' or a list of test names. The default is 'all'.
    maxlag : int, optional
        Maximum time lag. The default is 0.
    kw_randphase : dict, optional
        Extra keywords for Fourier transform test. The default is {}.
    kw_bstrap :  dict optional
        Extra keywords for bootstrap test. The default is {}.
    kw_twin : dict, optional
        Extra keywords for twin test. The default is {}.
    r_tts : int or function, optional
        Either a truncation radius or a function that picks one. The default is choose_r.
    r_naive : int or function, optional
        Either a truncation radius or a function that picks one. The default is choose_r.
    sig_test : function, optional
        Function to use to run the test. This affects how delays are accounted for.
        The default is sig_test_good.
    tts_test : function, optional
        Function to use to run the truncated time shift test.
        This affects how delays are accounted for.

    Returns
    -------
    pvals : dict
        Dictionary of pvalues for each test.

    """
    pvals = {};
    if test_list == 'all':
        test_list = ['randphase', 'bstrap', 'twin', 'tts', 'tts_naive', 'circperm','perm']
    if type(r_tts) is type(1):
        r = r_tts
        def r_tts(x):
            return r
    if type(r_naive) is type(1):
        r_ = r_naive
        def r_naive(x):
            return r_
    #trimming
    k_start,k_end = trim_periodic_data(y);
    xtrim = x[k_start:k_end+1]
    ytrim = y[k_start:k_end+1]
    #run tests
    #get random shuffle significance
    if 'perm' in test_list:
            pvals['perm'] = sig_test(x,y,statistic,get_perm_surrogates,maxlag);
	# get stationary bootstrap significance
    # (use trimmed data)
    if 'bstrap' in test_list:
    	    pvals['bstrap'] = sig_test(xtrim,ytrim,statistic,get_stationary_bstrap_surrogates,maxlag);
	# get twin significance
    if 'twin' in test_list:
            pvals['twin'] = sig_test(x,y,statistic,get_twin_surrogates,maxlag);
	# get TTS significance
    if 'tts' in test_list:
            r = r_tts(x.size);
            B = tts_test(x, y, r, statistic,maxlag);
            # make adjustment to pvalue so test is exactly valid [2]
            pvals['tts'] = B * (2 * r + 1) / (r + 1)
	# get naive TTS significance
    if 'tts_naive' in test_list:
            r = r_naive(x.size)
            pvals['tts_naive'] = tts_test(x, y, r, statistic, maxlag)
    #tts multishift
    if 'tts_multishift' in test_list:
        r = r_multishift(x.size, maxlag);
        lag_grid = np.arange(-maxlag,maxlag+1,1);
        pvals['tts_multishift'] = multishift_sigf(x,y,'tts', statistic, lag_grid, r);
	# get random phase significance
    # (use trimmed data)
    if 'randphase' in test_list:
            pvals['randphase'] = sig_test(xtrim,ytrim,statistic,get_iaaft_surr,maxlag);
    # get circular permutation significance
    # (use trimmed data)
    if 'circperm' in test_list:
            pvals['circperm'] = sig_test(xtrim,ytrim,statistic,get_circperm_surrogates,maxlag);
    return pvals

def benchmark_stats(x,y,test_list='all',maxlag=5):
    """
    Run a given list of tests for all statistics. Includes analytical tests.
    For directional statistics, only use the target of prediction to make surrogates.

    Parameters
    ----------
    x : numpy.array
        1D time series of length N.
    y : numpy.array
        1D time series of length N.
    test_list : list or string, optional
        Either 'all' or a list of test names. The default is 'all'.
    maxlag : int, optional
        Maximum time lag. The default is 5.

    Returns
    -------
    pvals : dict
        A dictionary of p-values for each statistic / analytical test.

    """
    #Granger causality y->x (uses y as surrogate template)
    pvals_granger = get_sigf(x,y,granger_stat_yx,test_list=test_list,maxlag=maxlag);
    #Granger causality x->y (uses x as surrogate template)
    pvals_granger_rev = get_sigf(y,x,granger_stat_yx,test_list=test_list,maxlag=maxlag);
    #Local similarity
    pvals_lsa = get_sigf(x,y,lsa_new_delay,test_list=test_list,maxlag=maxlag);
    #Pearson correlation
    pvals_pcorr = get_sigf(x,y,pcorr_strength,test_list=test_list,maxlag=maxlag);
    #Cross map skill y->x (uses y as surrogate template)
    pvals_ccm = get_sigf(x,y,ccm_statistic_yx,test_list=test_list,maxlag=maxlag);
    #Cross map skill x->y (uses x as surrogate template)
    pvals_ccm_rev = get_sigf(y,x,ccm_statistic_yx,test_list=test_list,maxlag=maxlag);
    #Mutual information
    pvals_mutual_info = get_sigf(x,y,mutual_info,test_list=test_list,maxlag=maxlag);
    
    #to account for lags in the analytical tests, we run them at each time lag,
    #pick the lowest p-value and multiply by the number of lags we tested.
    
    #parametric Pearson correlation test
    pval_parametric = {"pearson":(2*maxlag+1)*np.min(delay_scan(x,y,modified_ttest,maxlag=maxlag)[2])};
    #Granger causality y->x F-test
    pval_granger_nosurr = {"granger_nosurr":(2*maxlag+1)*np.min(delay_scan(x,y.reshape(-1,1), \
                        granger_stat_yx,maxlag=maxlag,kw_statistic={'pval':True})[1])};
    #Granger causality x->y F-test
    pval_granger_nosurr_rev = {"granger_nosurr":(2*maxlag+1)*np.min(delay_scan(x,y.reshape(-1,1), \
                        granger_stat_xy,maxlag=maxlag,kw_statistic={'pval':True})[1])};
    #Analytical local similarity test
    pvals_lsa_dd = {"lsa":(2*maxlag+1)*np.min(delay_scan(x,y.reshape(-1,1),dd_lsa,maxlag=maxlag)[2])};

    return {'granger_y->x':pvals_granger, \
            'granger_x->y':pvals_granger_rev, \
            'lsa':pvals_lsa, \
            'pearson':pvals_pcorr, \
            'ccm_y->x':pvals_ccm, \
            'ccm_x->y':pvals_ccm_rev, \
            'mutual_info':pvals_mutual_info, \
            'pcc_param':pval_parametric, \
            'granger_param_y->x':pval_granger_nosurr, \
            'granger_param_x->y':pval_granger_nosurr_rev, \
            'lsa_data_driven':pvals_lsa_dd};
