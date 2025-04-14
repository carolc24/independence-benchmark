"""
Main function and helper functions for convergent cross mapping

Main author: Caroline Cannistra (carolc24@uw.edu)
Contributing authors: Alex Yuan, Linh Hoang

References: 
[1]      George Sugihara et al., Detecting Causality in Complex Ecosystems. Science 338,496-500(2012).DOI:10.1126/science.1227079
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.spatial import distance_matrix
from pandas import DataFrame

#pearson correlation
#from scipy.stats
def pcorr(x,y):
    return pearsonr(x,y)[0]

#Calculate cross map skill
#this function can be used to see if cross map skill "converges"
#with increasing library sizes. in this paper we don't check for convergence.
def ccm_loocv(data, embed_dim, tau=1, lib_sizes=[None], replace=False,
              n_replicates=1, weights='exp', pred_lag=0, score=pcorr):
    """
    Args:
        data (array): array with n rows and m+1 columns, where n is the number
            of time points and m is the number of putative causer variables. The
            first column is the putative causee
        embed_dim (int): embedding dimension
        tau (int): delay for delay embedding
        lib_sizes (iterable): list of library sizes. Default is maximum library size
        replace (False): replacement for libraries chosen from sizes below maximum
        n_replicates (int): number of replicate runs
        weights (string): weighting function. Choices are 'exp' or 'uniform'
        pred_lag (int): time lag between predictor and predictee
        score (function): method to score correlation, pcorr by default
    Returns:
        Pandas DataFrame. The column 'causer' denotes the index of the putative
            causer
    """
    
    X, _ = setup_problem(data, embed_dim=embed_dim, tau=tau, pred_lag=pred_lag)
    n_y = data.shape[-1] - 1
    Y = [setup_problem(data[:,[0,i+1]], embed_dim=embed_dim, tau=tau, pred_lag=pred_lag)[1] for i in range(n_y)]
    # STEP ONE: get true neigbor indices
    KX = distance_matrix(X,X)
    np.fill_diagonal(KX, np.inf) # not allowed to train on self
    (n, embed_dim) = X.shape
    RX = np.zeros_like(KX)
    for loc in range(n):
        RX[loc,:] = np.argsort(KX[loc])                # sort vec indices by dist to loc
    all_xnbrs = np.zeros([n, embed_dim+1])
    result = []
    for y_idx, y in enumerate(Y):
        for lib_size in lib_sizes:
            for rep in range(n_replicates):
                if lib_size is None:
                    all_xnbrs = RX[:, :embed_dim+1]
                else:
                    nbr_rankings = np.sort(my_random_choice(n-1, size=lib_size, n=n, replace=replace).T)[:,:embed_dim+1]
                    for loc in range(n):
                        all_xnbrs[loc,:] = RX[loc,:][nbr_rankings[loc,:]]
                # STEP TWO: compute observed cross-map skill
                if weights == 'uniform':
                    yhat = np.mean(y[all_xnbrs.astype(np.int)], axis=1) # compute observed xmap skill
                elif weights == 'exp':
                    yhat = [np.sum( y[all_xnbrs[loc,:].astype(np.int)] *
                                    get_weights(KX[loc,all_xnbrs[loc,:].astype(np.int)])
                                  )
                            for loc in range(n)]
                    yhat = np.array(yhat)
                else:
                    raise ValueError('weights must be `exp` or `uniform`.')
                result.append([lib_size, y_idx, score(y,yhat)])
    result = np.array(result)
    return DataFrame(result, columns=['lib_size', 'causer', 'score'])

def get_weights(distances):
    """Performs weighting in KNN for Simplex projection

    Args:
      * distances (array): Array of distances

    Returns:
      * weights
    """
    distances = np.array(distances, copy=True).astype(np.float)
    if len(distances.shape) == 1:
        distances = distances.reshape(1,-1)
    if np.any(distances == 0): # deal with zeros
        distances[distances != 0] = np.inf
        distances[distances == 0] = 1
    min_dist = np.min(distances, axis=1).reshape(-1,1)
    u = np.exp(-distances / min_dist)
    weights = u / np.sum(u, axis=1).reshape(-1,1)
    return weights

def my_random_choice(a, size, n, replace=False):
    """
    Args:
        a (int): number of items to choose from
        size (int): number of items to choose on each replicate trial
        n (int): number of trials
        replace (bool): choose with or without replacement

    Returns:
        an (size x n) matrix where each column is a trial
    """
    if replace:
        return np.random.choice(a, size=[size, n], replace=True)
    else:
        return np.argsort(np.random.random([a, n]), axis=0)[:size, :]

def setup_problem(data, embed_dim, tau, pred_lag=0):
    """Prepares a "standard matrix problem" from the time series data

    Args:
        data (array): A 2d array with two columns where the first column
            will be used to generate features and the second column will
            be used to generate the response variable
        embed_dim (int): embedding dimension (delay vector length)
        tau (int): delay between values
        pred_lag (int): prediction lag
    """
    x = data[:,0]
    y = data[:,1]
    feat = []
    resp = []
    idx_template = np.array([pred_lag] + [-i*tau for i in range(embed_dim)])
    for i in range(x.size):
        if np.min(idx_template + i) >= 0 and np.max(idx_template + i) < x.size:
            feat.append(x[idx_template[1:] + i])
            resp.append(y[idx_template[0] + i])
    return np.array(feat), np.array(resp)

def choose_embed_params(ts, ed_grid=np.arange(1,9), tau_grid=np.arange(1,9), weights='exp', score=pcorr):
    """Performs a grid search to find the best delay embedding for a time series
       by testing the cross map skill of ts on itself with a prediction lag of 1

    Args:
        ts (array): 1d array, time series
        ed_grid (array): choices for embedding dimension
        tau_grid (array): choices for delay between values
        weights (string): weighting method for CCM. choices are "uniform" and "exp"
        score (function): performance scoring method for CCM. default is pcorr
    """
    data = np.zeros([len(ts), 2])
    data[:,0] = ts
    data[:,1] = ts
    result = []
    #grid search
    for embed_dim in ed_grid:
        for tau in tau_grid:
            #skip if params don't work for library size
            if (ts.shape[0] - 3 <= tau * (embed_dim - 1)):
                continue;
            #find cross map skill of the time series on itself with prediction lag
            rho = ccm_loocv(data, embed_dim=embed_dim, tau=tau, pred_lag=1,
                                 weights=weights, score=score).score.values.item()
            result.append([embed_dim, tau, rho])
    result = np.array(result)
    #choose params with highest score
    winner_idx = np.argmax(result[:,2])
    return result[winner_idx, 0].astype(int), result[winner_idx, 1].astype(int)

"""
n_vecs = ts_length - tau * (embed_dim - 1)
ts_length = n_vecs + tau * (embed_dim - 1)
"""

def prepare_climate_series(x, y, x_start, embed_dim, tau):
    """ When x is much longer than y, this function prepares data so that the
    number of vectors can be independent of the delay parameters.

    x (numpy array): a long time series
    y (numpy array): a short time series
    x_start (int): y[0] is going to be aligned to x[x_start]
    embed_dim (int): embedding dimension for ccm
    tau (int): delay vector lag for ccm
    """
    n_vecs = y.size
    vec_start = x_start - tau * (embed_dim - 1)
    x_end = x_start + n_vecs
    data = np.zeros([x_end - vec_start,2])
    data[:,0] = x[vec_start:x_end]
    data[tau * (embed_dim - 1):,1] = y
    return data

#find cross map skill with increasingly larger time series
#and print them. we don't use this
def ccm_converge_assay(data):
    N=data.shape[0];
    chunk=int(np.floor(N/20));
    min_size = N-chunk*19
    
    #choose embed params that work for all library sizes    
    embed_dim,tau = choose_embed_params(data[:min_size,0]);
    
    t = np.arange(min_size, N+1, chunk);
    cms = np.zeros(t.shape)
    for i in range(t.size):
        m = t[i];
        cms[i] = ccm_loocv(data[:m],embed_dim,tau)['score']
        
    #measure of convergence
    return kendalltau(t,cms)[1]