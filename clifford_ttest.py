"""
Author: Alex Yuan

References:

[1] Clifford, P., Richardson, S., & Hemon, D. (1989). Assessing the significance of the correlation between two spatial processes. Biometrics, 123-134.

[2] Dutilleul, P., Clifford, P., Richardson, S., & Hemon, D. (1993). Modifying the t test for assessing the correlation between two spatial processes. Biometrics, 305-314.

[3] Pyper, B. J., & Peterman, R. M. (1998). Comparison of methods to account for autocorrelation in correlation analyses of fish data. Canadian Journal of Fisheries and Aquatic Sciences, 55(9), 2127-2140.
"""
import numpy as np
from scipy.stats import t as students_t

def acov(x, max_lag=None):
    """Sample autocovariance of a time series (Eq 2.5 of Ref 1)
    Args:
     * x (1d array): a time series with regularly spaced time points
     * max_lag (int): maximum lag for computing autocovariance

    Returns:
     * covariance estimate (C_hat in Ref 1)
     * cardinality of ordered pairs at each lag (N_k in Ref 1)
     * lags (indexed by k in Ref 1)
    """
    # process max_lag argument
    if max_lag == None:
        n_strata = x.size
    else:
        n_strata = max_lag + 1
    # get sample mean
    xbar = np.mean(x)
    # initialize data structures to contain results
    cov = np.zeros(n_strata)
    card= np.zeros(n_strata)
    lag = np.arange(n_strata)
    # get covariance and cardinality of each stratum
    for i in range(n_strata):
        cov[i] = np.mean( (x[lag[i]:]-xbar)*(x[:x.size-lag[i]]-xbar) )
        card[i] = x[lag[i]:].size
        if i > 0:
            card[i] *= 2
    return cov, card, lag

def pcorr(x, y):
    """sample Pearson correlation (Eq 2.1 of Ref 1)
    """
    xbar = np.mean(x)
    ybar = np.mean(y)
    s_xy = np.mean((x-xbar) * (y-ybar))
    return s_xy / (np.sqrt(svar(x)) * np.sqrt(svar(y)))

def svar(x):
    """Sample variance computed as in the paragraph below Eq 2.1 of Ref 1:
    """
    xbar = np.mean(x)
    return np.mean((x-xbar)**2)

def rhovar_clifford(x, y, max_lag=None):
    """Estimated variance of the correlation coefficient between x and y (Eq 2.9 in Ref 1)

    Args:
     * x (1d array): a time series with regularly spaced time points
     * y (1d array): a time series with regularly spaced time points; same length as x
     * max_lag (int): maximum lag for computing autocovariance

     Returns:
      * estimated variance of correlation coefficient
    """
    cov_x, card_x, lag_x = acov(x, max_lag=max_lag)
    cov_y, card_y, lag_y = acov(y, max_lag=max_lag)
    sx = svar(x)
    sy = svar(y)
    return np.sum(cov_x * cov_y * card_x) / (x.size**2 * sx * sy)

def acov_matrix(x, max_lag=None):
    """Sample autocovariance of a time series, but as a matrix
    Args:
     * x (1d array): a time series with regularly spaced time points
     * max_lag (int): maximum lag for computing autocovariance.

    Returns:
     * autocovariance matrix. Matrix is n-by-n, where n is the length of x.
       Entries that refer to lags beyond max_lag are set to zero.
    """
    cov, card, lag_grid = acov(x, max_lag=max_lag)
    sig = np.zeros([x.size, x.size])
    lag_matrix = np.abs(np.arange(x.size).reshape(-1,1) - np.arange(x.size).reshape(1,-1))
    for lag in range(cov.size):
        sig[lag_matrix==lag] = cov[lag]
    return sig

def rhovar_dutilleul(x, y, max_lag=None):
    """Estimated variance of the correlation coefficient between x and y (Eq 3.5 in Ref 2)

    Args:
     * x (1d array): a time series with regularly spaced time points
     * y (1d array): a time series with regularly spaced time points; same length as x
     * max_lag (int): maximum lag for computing autocovariance

     Returns:
      * estimated variance of correlation coefficient
    """
    n = x.size
    J = np.ones([n,n])
    I = np.identity(n)
    B = (1/n) * (I - (1/n) * J)
    sig_x = acov_matrix(x, max_lag=max_lag)
    sig_y = acov_matrix(y, max_lag=max_lag)
    return np.trace(B @ sig_x @ B @ sig_y) / (np.trace(B @ sig_x) * np.trace(B @ sig_y))

def modified_ttest(x, y, max_lag=None, rhovar_estimator='dutilleul',
                   prohibit_negative_rhovar=True):
    """Perform a two-tailed autocorrelation-aware t-test of
    correlation between x and y

    Args:
     * x (1d array): a time series with regularly spaced time points
     * y (1d array): a time series with regularly spaced time points; same length as x
     * max_lag (int): maximum lag for computing autocovariance. Ref 3 recommends n/4 or n/5.
     * rhovar_estimator (string): Estimator for variance of sample correlation
       coefficient. Options are 'dutilleul' (the method of Ref 2) or 'clifford' (the method of Ref 1).
     * prohibit_negative_rhovar (bool): By default, if rhovar is negative, it is
       replaced by 1/x.size, which corresponds to the case without autocorrelation
       as recommended by Ref 1. Set this parameter to False to disable this behavior.

    Returns:
      * sample pearson correlation coefficient
      * two-tailed p-value
    """
    if max_lag==None:
        max_lag = int(x.size/4);
    rho = pcorr(x, y)
    if rhovar_estimator == 'clifford':
        rhovar = rhovar_clifford(x, y, max_lag=max_lag)
    elif rhovar_estimator == 'dutilleul':
        rhovar = rhovar_dutilleul(x, y, max_lag=max_lag)
    if (rhovar < 0) and (prohibit_negative_rhovar):
        rhovar = 1/x.size
    n_eff = 1 + 1/rhovar
    t_stat = rho * np.sqrt((n_eff - 2) / (1 - rho**2))
    p = 2*(1-students_t.cdf(np.abs(t_stat), df=n_eff-2))
    return np.array([rho, p])
