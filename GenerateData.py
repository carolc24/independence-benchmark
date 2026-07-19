#%% Import libraries
import logging
log = logging.getLogger(__name__)
# import pandas as pd
import scipy as sp
from scipy.integrate import solve_ivp
import numpy as np
import random

# Lotka Volterra
def lotkaVolterra(t,y,mu,M):
    return y * (mu + M @ y);  

def lotkaVolterraSat(t,y,mu,M,K):    
    intx_mat = M/(K + y);
    lv = y * (mu + intx_mat @ y);
    return lv;

def generate_lv(dt_s, N, s0, mu, M, noise, noise_T, 
                time_skip = 250, 
                fn = lotkaVolterra, measurement_noise = 0, scale_noise=False): # ,intx="competitive"
    log.info('Generating generalised Lotka-Volterra model')
    dt=0.05; # integration step
    lag = int(time_skip/dt) # int(150/dt) # 30000
    sample_period = int(np.ceil(dt_s / dt)); 
    obs = sample_period * N;
    n = len(mu)
    s = np.zeros((lag + obs + 1, n))
    
    args = (mu,M);
    s[0] = s0
    
    
    # a = 0.025
    for i in range(lag + obs):
        soln = solve_ivp(fn,[0,dt],s[i],args=args)
        # soln = solve_ivp(fn,[0,dt],s[i],args=(mu+a*(s[i] < 0.05),M)) # a = 0.2
        '''
        For growth boost
        soln = solve_ivp(lotkaVolterra,[0,dt],s[i],args=(mu+a*(s[i] < 0.05),M)) # a = 0.2
        *Notes: not mentioned in paper
        '''
        if scale_noise:
            eps_ = np.sqrt(np.maximum(soln.y[:, -1], 0.0 )) # recommended = nitesh and akshit, makes extinct species stay extinct # eq: x_i = x_i + x* + sqrt(x_i)*eps_i with eps_i~Normal distribution
        else:
            eps_ = 1 # eps = noise*np.random.randn(n)*np.random.binomial(1,dt/noise_T,size=n); # process noise/external perturbation = allow migration over time.
        eps = eps_*noise*np.random.randn(n)*np.random.binomial(1,dt/noise_T,size=n)
        
        # s[i+1] = soln.y[:,-1] + eps; # print(s[i+1])
        # s[i+1][np.where(s[i+1] < 0)] = 0;
        nxt = soln.y[:, -1] + eps
        nxt[nxt<10e-8] = 0
        # nxt[nxt<0] = 0 
        s[i+1] = nxt

    x = s[lag:lag+obs:sample_period,]; 
    x = x + measurement_noise * np.random.randn(*x.shape) 

    # return [x[:,_] for _ in [0,1]] # for 2 species
    return x # for multispecies
    # return s # checking the full dynamic
#%% Caroline equally competitive matrix generation
def im_equalinterxn_M(S, diag=-0.4, offdiag=-0.05, dtype=float):
    """
    Equal-interaction gLV matrix.
    - diag: self-interaction M_ii (should be negative for self-limitation)
    - offdiag: interaction M_ij for i!=j (negative = competition, positive = mutualism)
    """
    M = np.full((S, S), offdiag, dtype=dtype)
    np.fill_diagonal(M, diag)
    return M
#%% Vano 4-species strange attractor
mu_vano = np.array([1, 0.72, 1.53, 1.27])
M_vano = np.array([[1,    1.09, 1.52, 0   ],
                   [0,    1,    0.44, 1.36], 
                   [2.33, 0,    1,    0.47], 
                   [1.21, 0.51, 0.35, 1   ]])
'''
    gLV form:
        dy/dt = y ⊙ (mu + M y)
    ----------
    Vano form:
        dy/dt = r ⊙ y ⊙ (1 - A y)
        - mu = r
        - M = −diag(r) · A
    ----------
    | Code              | What it does        | Mathematical meaning        |
    |-------------------|---------------------|-----------------------------|
    | r[:, None] * A    | Scales rows of A    | (r_i · A_ij)                |
    | A * r             | Scales columns of A | (A_ij · r_j)                |
    | np.diag(r) @ A    | Scales rows of A    | (r_i · A_ij)                |
    | A @ np.diag(r)    | Scales columns of A | (A_ij · r_j)                |
'''
def scale_offdiag(A, s):
    '''
    A : is the interaction matrix M but in Vano annotation...
    s : coupling-scaling bifurcation parameter
    ----------
    Ascaled : A_{ij} -> s*A_{ij} with i!=j

    '''
    Ascaled = s * A.copy()
    np.fill_diagonal(Ascaled, np.diag(A))
    # D = np.diag(np.diag(A))
    # return s * A + (1 - s) * D
    return Ascaled

def convertAvano2M(A, mu):
    '''
    M : interaction matrix
    mu : base growth rate
    '''
    return -A * np.expand_dims(mu, 1) # M = -(r[:, None] * A)
#%% Multiple species
from numpy.random import default_rng

def initial_conditions_s0(S):
    # S : number of species in community. Returns random s0 of size S
    # in Polo codes, the makepool and makeplate contribute to this function
    # in this study, I don't propagate the culture so I'm randomly generating initial condition
    return np.random.uniform(0.0, 1.0, size=S)

def intrinsic_growth_vector_mu(S, val=None, rng=None):
    """
    Generate intrinsic growth rate vector mu.

    Parameters
    ----------
    S : int
        Number of species.
    val : None | float | array-like | "random"
        - None       → mu_i = 1 for all i
        - float      → mu_i = val for all i
        - array-like → user-specified mu (length must be S)
        - "random"   → random positive mu_i (guaranteed non-zero)
    rng : np.random.Generator or None
        Optional RNG for reproducibility.

    Returns
    -------
    mu : np.ndarray, shape (S,)
    """
    rng = np.random.default_rng() if rng is None else rng

    if val is None:
        # Default: all ones (your current behaviour)
        mu = np.ones(S)
    elif isinstance(val, (int, float)):
        # Constant growth rate
        mu = np.full(S, float(val))
    elif isinstance(val, str):
        # Random but strictly positive (avoid zero-growth species)
        mu = rng.uniform(low=0.5, high=1.5, size=S)
    return mu

#%%% Polo Matlab extracted ver - May's work
'''
    Construct interaction matrix M.
    May's ecological stability theory. For a GLV or random large ecological network, if interaction coefficients scale like 1/sqrt(S), the system has well-defined stability threshold
    ----------
    Idea of May:
        - Simple question: what happens to stability when a system gets large and complex?
        - He considered: S species, random interactions, no structure, asked when equilibria are typically stable
    ----------
    Why scale by squrt(S)?
        - If naively draw A_{ij} ~ Normal(mu, sigma^2)
        Sum(A_{ij}Nj) would grow as S increases, the dynamics blows up
        - So May introduced scaling purely for mathematical reasons
            sigma_eff = sigma/squrt(S)
        - These parameters existed because May was studying random matrix spectra, not ecological realism
    ----------
        dy/dt = y ⊙ (K - y - Ay) = y ⊙ (K-(I+A)y)
    Normalised logistic form of gLV
        dy/dt = y ⊙ (1 - (I+A) @ y)
        - A_{ij} with i!=j, the diagonal has been taken out as -y = -Iy
        - mu = K = 1 (intrinsic_growth_vector_mu())
'''
def im_may_M(S, meanmu=30.0, sigma=4.0, gamma=-0.5, rng=None):
    """
    ----------
    S : int, number of species (S).
    meanmu : float, mean interaction parameter (unnormalised).
    sigma : float, std dev of interaction parameter (unnormalised).
    gamma : float, correlation between a_ij and a_ji.
    rng : np.random.Generator or None, Random generator for reproducibility.
    -------
    M : (S, S) numpy array, NEGATED Interaction matrix A.
    """
    rng = default_rng() if rng is None else rng

    mean_scaled = meanmu / S
    std_scaled  = sigma / np.sqrt(S)

    Mean  = np.array([mean_scaled, mean_scaled])
    Sigma = np.array([[std_scaled**2, gamma * std_scaled**2],
                      [gamma * std_scaled**2, std_scaled**2]])

    # sample S*S correlated pairs (a_ij, a_ji)
    R = rng.multivariate_normal(Mean, Sigma, size=(S * S))
    R1 = R[:, 0].reshape(S, S)
    R2 = R[:, 1].reshape(S, S)

    A1u = np.triu(R1, 1)        # upper triangle (excluding diag)
    A2u = np.triu(R2, 1)        # another upper triangle → used for lower

    A  = A1u + A2u.T + np.eye(S)  # symmetric-ish + self = 1
    return -A

#%%% Akshit multistability
'''
    + symmetric matrix can not lead to chaos.
    + https://arxiv.org/pdf/2511.06697
    + how easy it is to know the threshold for stochasticity?
        + Akshit says noone knows that so we have to tune the schochasticity in slowly
    + non zeros entries will create the same outcome as discussed
    + what we observed from the 50 species seed(0) is chaos
    ----------
        dy/dt = y ⊙ (K - y - Ay) = y ⊙ (K-(I+A)y)
    The paper consider strong interaction regime: both meanmu and sigma are finite constants that do not scale with S number of species
        - contrast to May's scaling
        - fix K_i = 1
        - boundary for multistability
            sigma = frac{-mu+1}{squrt{2S}}
    ----------
    Multistability boundary
        \sigma=\frac{1-\mu}{\squrt{2S}}
'''
def im_symmetric_M(S, meanmu=0.5, sigma=0.3, rng=None):
    """
    S : int, number of species (S).
    meanmu : float, positive, mean interaction parameter (unnormalised).
    sigma : float, positive, std dev of interaction parameter (unnormalised).
    rng : np.random.Generator or None, Random generator for reproducibility.
    -------
    M : (S, S) numpy array, NEGATED Interaction matrix A.
    """
    rng = default_rng() if rng is None else rng
    A = rng.normal(loc=meanmu, scale=sigma, size=(S,S))
    while np.any(A == 0.0):
        loc = (A == 0.0)
        A[loc] = rng.normal(loc=meanmu, scale=sigma, size=loc.sum())
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    return -A

def multistability_crit(meanmu, S):
    """
    Patro et al. (Eq. 2): multistability boundary for symmetric random GLV.
        sigma_c = (1 - mu) / sqrt(2S)
    """
    return(1-meanmu)/np.sqrt(2.0*S)