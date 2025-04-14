# -*- coding: utf-8 -*-
"""
Correlation statistics used by dependence tests.
Main Author: Caroline Cannistra (carolc24@uw.edu)
Contributing Author: Alex Yuan

References:
    [1] Quansong Ruan and others, Local similarity analysis reveals unique associations among marine bacterioplankton species and environmental factors, Bioinformatics, Volume 22, Issue 20, October 2006, Pages 2532–2538, https://doi.org/10.1093/bioinformatics/btl417
    [2] Li C. Xia and others, Efficient statistical significance approximation for local similarity analysis of high-throughput time series data, Bioinformatics, Volume 29, Issue 2, January 2013, Pages 230–237, https://doi.org/10.1093/bioinformatics/bts668
    [3] Zhang, F., Sun, F. & Luan, Y. Statistical significance approximation for local similarity analysis of dependent time series data. BMC Bioinformatics 20, 53 (2019). https://doi.org/10.1186/s12859-019-2595-x
    [4] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, et. al. Scikit-learn: Machine Learning in Python. J. Mach. Learn. Res. 12, null (2/1/2011), 2825–2830.
    [5] Seabold, Skipper & Perktold, Josef. (2010). Statsmodels: Econometric and Statistical Modeling with Python. Proceedings of the 9th Python in Science Conference. 2010. 
    [6] Donald W. K. Andrews. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation. Econometrica, 59(3), 817–858. https://doi.org/10.2307/2938229
    
"""
#dependencies
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests as granger
from statsmodels.tsa.api import VAR
#our code
import ccm
from clifford_ttest import modified_ttest

#pearson correlation statistic
#x: Nx1 vector
#y: NxP matrix
def pcorr_strength(x,y):
    M = np.zeros([x.size, y.shape[1] + 1])
    M[:,0] = x
    M[:,1:] = y
    cov = np.cov(M.T)
    cov_xy = cov[0,:]
    std_y = np.sqrt(np.diag(cov))
    std_x = std_y[0]
    rho = cov_xy / (std_y * std_x)
    return np.abs(rho[1:]) #result is always positive

#normalize data for LSA
#x: Nx1 vector
def norm_transform(x):
    #returns points from a gaussian distribution
    #based on the rank of the original time point
    return stats.norm.ppf((stats.rankdata(x))/(x.size+1.0))

#calculate local similarity (From [1])
#x: Nx1 vector
#y: NxP matrix
#D: max delay. in this paper this is always set to 0.
def lsa_delay(x,y_array,D=0):

    n = x.size # number of time points
    x = np.copy(norm_transform(x)) # normalize x
    score_P = np.zeros((y_array.shape[1])); # S+ for each X,Y pair
    score_N = np.zeros((y_array.shape[1])); # S- for each X,Y pair
    
    #loop through each row of y
    for k in range(y_array.shape[1]):
        y = np.copy(norm_transform(y_array[:,k])); # normalize y
        P = np.zeros((n+1,n+1)) # 2D array of cumulative associations
        N = np.zeros((n+1,n+1))
        #loop through time points
        for i in range(x.size):
            for j in range(x.size):
                if np.abs(i - j) <= D:
                    #count positive and negative associations at different time lags
                    P[i+1][j+1] = np.max([0, P[i][j] + x[i] * y[j]])
                    N[i+1][j+1] = np.max([0, N[i][j] - x[i] * y[j]])
        #LS = max S+ or S- divided by time series length
        score_P[k] = np.max(P) / n;
        score_N[k] = np.max(N) / n;
    sign = np.sign(score_P - score_N); # LS can be positive or negative...
    return np.max([score_P, score_N],axis=0); # ...but we only want positive values

#mutual info statistic
#from [4]
#x: Nx1 vector
#y: NxP matrix
def mutual_info(x,y):
	return mutual_info_regression(y, x, n_neighbors=3).flatten()

#cross map skill statistic
#measuring causal effect of y on x
#use x (1d) to predict y (can be 2d)
def ccm_statistic_yx(x,y):
    #choose embedding params for x
    embed_dim, tau = ccm.choose_embed_params(x);
    #use those params to calculate cross map skill
    #for each row of y on x
    return np.array(ccm.ccm_loocv(np.vstack((x,y.T)).T, embed_dim, tau)['score']);

#cross map skill
#measuring causal effect of x on y
#use y (can be 2d) to predict x (1d)
#this is slow so we only use this method 
#to test x as a surrogate template
def ccm_statistic_xy(x,y):
    N,P = y.shape;
    if (P == 1): # only one time series in y
        #only need one embedding
        embed_dim, tau = ccm.choose_embed_params(y.reshape(-1));
        return np.array(ccm.ccm_loocv(np.vstack((y.T,x)).T, embed_dim, tau)['score']);
    else:
        cms = np.zeros(N);
        #we need a new embedding for each y
        #so we loop through them
        for i in range(N):
            embed_dim, tau = ccm.choose_embed_params(y[:,i].T);
            cms[i] = ccm.ccm_loocv(np.vstack((y[:,i].T,x)).T, embed_dim, tau)['score'];
        return cms;
    
#granger causality statistic
#function taken from [5]
#test causal effect of y on x
#use history of y (can be 2d) to predict x (1d)
#pval: boolean, if True return the pval from the F-test instead of the F-statistic
def granger_stat_yx(x,y,pval=False):
    t,N = y.shape;
    if (N == 1):
        data = np.vstack((x,y.T)).T;
        
        #use autogressive model 
        #and akaike's information criterion (AIC) to pick lag
        model = VAR(data);
        results = model.fit(maxlags=15,ic='aic');
        maxlag=np.max([results.k_ar, 1])
        
        if pval:
            return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        else:
            return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
    else:
        result = np.zeros(N);
        for i in range(N):
            data = np.vstack((x,y[:,i].T)).T;
            #use autogressive model 
            #and akaike's information criterion (AIC) to pick lag
            model = VAR(data);
            res = model.fit(maxlags=15,ic='aic');
            maxlag = np.max([res.k_ar, 1])
            if pval:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
            else:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0];
        return result;
    
#granger causality statistic
#measure causal effect of x on y
#use history of x (1d) to predict y (can be 2d)
#use aic to pick lag
def granger_stat_xy(x,y,pval=False):
    t,N = y.shape;
    if (N == 1):
        data = np.vstack((y.T,x)).T;
        #use autogressive model 
        #and akaike's information criterion (AIC) to pick lag
        model = VAR(data);
        results = model.fit(maxlags=15,ic='aic');
        maxlag=np.max([results.k_ar, 1])
        if pval:
            return granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
        else:
            return np.array(granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0]);
    else:
        result = np.zeros(N);
        for i in range(N):
            data = np.vstack((y[:,i].T,x)).T;    
            #use autogressive model 
            #and akaike's information criterion (AIC) to pick lag
            model = VAR(data);
            res = model.fit(maxlags=15,ic='aic');
            maxlag = np.max([res.k_ar, 1])
            if pval:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][1];
            else:
                result[i] = granger(data, [maxlag], verbose=False)[maxlag][0]['ssr_ftest'][0];
        return result;
        
#for analytical LSA test
#calculates autocorrelation at different lags
#equation 2 of [3]
#can also be found in appendix of this paper

#x: Nx1 vector
#acov: boolean, if True then calculate autocovariance instead of autocorrelation
def acorr_bj(x,acov=False):
    result = np.zeros(x.size)
    if acov:
        denomenator = x.size;
    else:
        denomenator = np.sum((x - np.mean(x))**2)
    for j in range(x.size):
        numerator = 0
        for t in range(x.size-j):
            numerator += (x[t] - np.mean(x)) * (x[t + j] - np.mean(x))
        result[j] = numerator / denomenator
    return result

#long run variance estimator
#eqs 3 and 4 from [3]
#uses a lag cutoff from [6]

#x: Nx1 vector, normalized
#y: Nx1 vector, normalized
def lrv(x,y):
    N = x.size;
    #elementwise product
    z = x * y.reshape(-1);
    #lag cutoff from eq 4 in zhang 2019
    resid = z - np.mean(z);
    rhohat = np.sum(resid[1:]*resid[:-1]) / np.sum(resid[:-1]**2)
    alphahat = 4*rhohat**2/(1-rhohat**2)**2
    bw = int(np.ceil(1.1447*(alphahat*N)**(1/3)));
    
    #calculate autocovariance at different lags
    x_acov = acorr_bj(x,acov=True).reshape(-1)
    y_acov = acorr_bj(y,acov=True).reshape(-1)
    z_acov = x_acov * y_acov;
    idx = np.arange(1,bw+1);
    weights = (bw - idx) / bw;
    
    #sum up weighted autocovariances
    #eq 3 of zhang 2019
    omega = z_acov[0] + 2*np.sum(weights * z_acov[1:bw+1])
    return omega;

#theoretical p value estimator for local similarity
#from [2]

#ls: scalar, local similarity score
#N: scalar, time series length
#var: scalar, estimated variance of ls
#D: scalar, max delay. always set to 0 here
def lsa_theo(ls, N, var, D):
    norm_ls = ls/np.sqrt(var*N);
    if (norm_ls == 0):
        tApprox_new = 1
    else:
        partial_sum = 0
        tApprox_new = 0
        i=1;
        tApprox_diff = 1;
        while (tApprox_diff > 1e-6):
            A = norm_ls**2;
            B = (2*i - 1)**2 * np.pi**2;
            threshold = (1/A + 1/B)*np.exp(-B/(2*A));
            partial_sum += threshold;
            tApprox_old = tApprox_new;
            tApprox_new = 1-8**(2*D+1)*partial_sum**(2*D+1);
            tApprox_diff = np.abs(tApprox_new - tApprox_old);
            i += 1;
    return tApprox_new;
    
#get pval from LSA with theoretical approximation 
#+ long run variance estimator
#from [3]
#x: Nx1 vector
#y: Nx1 vector
#D: max delay, always 0 here
def dd_lsa(x,y,D=0):
    N = x.size;
    #zhang's code uses non-normalized LS so we correct here
    sd = lsa_delay(x,y,D) * N;
    #normalize x and y
    x = np.copy(norm_transform(x));
    y = np.copy(norm_transform(y)).reshape(-1,1);
    #estimate long run variance
    #if we get an error we will use a backup method
    try:
        xOmegay = lrv(x,y.reshape(-1));
    except ValueError:
        xOmegay = np.nan;
    #backup method for calculating variance
    var = np.var(x)*np.var(y);
    #then use theoretical approximation for pval
    if (np.isnan(xOmegay)):
        approximation = lsa_theo(sd, N, var, D)
    else:
        approximation = lsa_theo(sd, N, xOmegay, D)
    #return LS and pval
    return np.array([sd / N, approximation]).reshape(-1);

