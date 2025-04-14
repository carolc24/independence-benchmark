# -*- coding: utf-8 -*-
"""
Generating time series data
Main author: Caroline Cannistra
Contributing authors: Alex Yuan, Linh Hoang

References:
    [1] Smale, S. On the differential equations of species in competition. J. Math. Biology 3, 5–7 (1976). https://doi.org/10.1007/BF00307854
    [2] Niehaus, L., Boland, I., Liu, M. et al. Microbial coexistence through chemical-mediated interactions. Nat Commun 10, 2052 (2019). https://doi.org/10.1038/s41467-019-10062-x

"""

import numpy as np
from scipy.integrate import solve_ivp

#generalized Lotka Volterra function
#args: t,y,mu,M
#t: time (scalar)
#y: 1xN vector of state variables
#mu: 1xN vector of baseline growth rates
#M: NxN matrix of interaction coefficients
#returns dydt, 1xN vector
def lotkaVolterra(t,y,mu,M):
    return y * (mu + M @ y);

#runs gLV simulation
#dt_s: sampling rate (1/sec)
#N: desired time series length
#noise: magnitude of process noise term
#measurement_noise: magnitude of measurement noise
#intx: string, describes type of system
def test_stats_glv(dt_s, N, noise, measurement_noise, intx="competitive_gaussian_noise"):
    dt=0.05; #integration step size
    mu = np.tile([0.7,0.7],2); # two sets of identical interacting pairs
    M = np.zeros((4,4));
    spike = 0; #increase in growth rate when pop density is low
    #set system parameters
    #corresponds to table 3 of the main text
    if (intx=="competitive_uniform_noise"):
        M[:2,:2] = [[-0.4,-0.5],[-0.5,-0.4]];
        M[2:,2:] = M[:2,:2];
        s0 = 5.*np.random.random(4); # random initial conditions
    if (intx=="competitive_gaussian_noise"):
        M[:2,:2] = [[-0.4,-0.5],[-0.5,-0.4]];
        M[2:,2:] = M[:2,:2];
        s0 = [2.,0.,2.,0.]; # high x1, low y1, high x2, low y2
    if (intx=="competitive_spiking"):
        M[:2,:2] = [[-0.4,-0.5],[-0.5,-0.4]];
        M[2:,2:] = M[:2,:2];
        s0 = [2.,0.,2.,0.]; # high x1, low y1, high x2, low y2
        spike = 0.2;
    if (intx=="predprey"): # pred-prey
        mu = np.tile([1.1,-0.4],2);
        M[:2,:2] = [[0.0,-0.4],[0.1,0.0]];
        M[2:,2:] = M[:2,:2];
        s0 = 2.*np.random.random(4); # random initial conditions
    #period of sampling for time series data
    sample_period = int(np.ceil(dt_s / dt));
    #let simulation run 150 sec before starting data collection
    lag = int(150/dt);
    #then run until we have desired number of data points
    obs = sample_period * N;

    #initialize 4 time series for simulation purposes only
    s = np.zeros((lag + obs + 1, 4))
    s[0] = s0; 
    #run for lag+obs steps
    for i in range(lag + obs):
        soln = solve_ivp(lotkaVolterra,[0,dt],s[i],args=(mu+spike*(s[i] < 0.4),M))
        if (intx=="competitive_uniform_noise"):
            eps = noise*dt*np.random.random(4); # uniform noise [0,noise]
        else:
            eps = noise*dt*np.random.randn(4); #gaussian noise, mu=0, sigma=noise
        s[i+1] = soln.y[:,-1] + eps;
        s[i+1][np.where(s[i+1] < 0)] = 0 # make sure we don't have negative y

    #add measurement noise
    s += measurement_noise*np.random.randn(s.size).reshape(s.shape);
    
    #actual time series subsampled from simulation data
    #measure true positive rates by testing x and y
    #measure false positive rates by testing x and yf
    x = s[lag::sample_period,0]
    y = s[lag::sample_period,1]
    yf = s[lag::sample_period,3]
    
    return [x,y,yf]
    
#diffeqs for chemically mediated interactions
#from Niehaus et al (2019)
def dSCdt(SC, num_spec, r0, K, alpha, beta, rho_plus, rho_minus):
    """
    Parameters:

    SC (array): an array of species and chemical abundances in which species
        are listed before chemicals
    num_spec (int): number of species
    r0 (2d numpy.array): num_spec x 1 array of intrinsic growth rates
    K (2d numpy.array): num_spec x num_chem array of K values
    alpha (2d numpy.array): num_chem x num_spec array of consumption constants
    beta (2d numpy.array): num_chem x num_spec array of production constants
    rho_plus (2d numpy.array): num_spec x num_chem array of positive influences
    rho_minus (2d numpy.array): num_spec x num_chem array of negative influences
    """

    S = np.reshape(SC[:num_spec], [num_spec,1])
    C = np.reshape(SC[num_spec:], [len(SC) - num_spec, 1])
    # compute K_star
    K_star = K + C.T
    # compute K_dd
    K_dd = rho_plus * np.reciprocal(K_star)
    # compute lambda
    Lambda = np.matmul(K_dd - rho_minus, C)
    # compute dS/dt
    S_prime = (r0 + Lambda) * S
    # compute K_dag
    C_broadcasted = np.zeros_like(K.T) + C
    K_dag = np.reciprocal(C_broadcasted + K.T) * C_broadcasted
    # compute dC/dt
    C_prime = np.matmul(beta - (alpha * K_dag), S)
    SC_prime = np.vstack((S_prime, C_prime))
    return SC_prime

#for chemically mediated interactions (add constant rsrc flux)
def sc_prime_rsrc(t, y, num_spec, r0, K, alpha, beta, rho_plus, rho_minus,r_flux):
    dy = dSCdt(y, num_spec, r0, K, alpha, beta, rho_plus, rho_minus);
    dy[-1] += r_flux;
    return np.reshape(dy, dy.size).tolist()

#generate time series data with chemically mediated interactions
def test_stats_niehaus(dt_s, N, noise, measurement_noise):
    dt=0.05;
    sample_period = int(np.ceil(dt_s / dt));
    lag = int(150/dt);
    obs = sample_period * N;
    
    num_spec = 2
    num_chem = 1
    r0 = np.array([[-1.6],[-1.6]])
    K = np.array([[5.0],
                  [5.0]])
    alpha = np.array([[4.0,4.0]])
    beta = np.array([[0.0,0.0]])
    rho_plus = np.array([[4.8],
                  [4.8]])
    rho_minus = np.zeros((num_spec,num_chem))
    r_flux = 4;
    
    params = (num_spec, r0, K, alpha, beta, rho_plus, rho_minus, r_flux);
    
    #we do 2 simulation loops for this one instead of running both reps together
    s = np.zeros((lag + obs + 1, num_spec + num_chem))
    s[0] = np.zeros((num_spec + num_chem))
    s[0,:2] = 1;
    
    #first rep
    for i in range(lag + obs):
        soln = solve_ivp(sc_prime_rsrc,[0,dt],s[i],args=params);
        s[i+1] = soln.y[:,-1] + noise*dt*np.random.random(num_spec + num_chem);
        s[i+1][np.where(s[i+1] < 0)] = 0

    s += measurement_noise*np.random.randn(s.size).reshape(s.shape)
    x = s[lag::sample_period,0].copy()
    y = s[lag::sample_period,1].copy()
    
    #second, independent rep
    for i in range(lag + obs):
        soln = solve_ivp(sc_prime_rsrc,[0,dt],s[i],args=params);
        s[i+1] = soln.y[:,-1] + noise*dt*np.random.random(num_spec + num_chem);
        s[i+1][np.where(s[i+1] < 0)] = 0
    
    s += measurement_noise*np.random.randn(s.size).reshape(s.shape)
    yf = s[lag::sample_period,1].copy() # different realization of y
    
    return [x,y,yf]

#function calls for each collection of time series we tested
#NOTE: each call generates one set of time series!
#to replicate our results, you will need to run each function many times.

#competitive gLV with uniform noise, N=200 (figure 2A-C)
#test_stats_glv(0.25, 200, 0.5, 0.,"competitive_uniform_noise")

#chemically mediated competition (figure 2D-F)
#test_stats_niehaus(0.25,200,0.5,0.)

#predator-prey (all except tts multishift) (figure 4)
#test_stats_glv(0.25,250,0.2,0.1,"predprey")

#predator-prey (multishift) (figure 4E)
#test_stats_glv(0.25,1000,0.2,0.1,"predprey")

#competitive gLV with gaussian noise, slow sampling (figure 5A-C)
#test_stats_glv(1.25,500,0.2,0.001,"competitive_gaussian_noise")

#competitive gLV with low density growth spiking (figure 5D-F)
#test_stats_glv(1.25,500,0.2,0.001,"competitive_spiking")

#competitive gLV with gaussian noise, fast sampling (figure S1)
#test_stats_glv(0.25,500,0.2,0.001,"competitive_gaussian_noise")