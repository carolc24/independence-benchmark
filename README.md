# independence-benchmark
Supplementary Code for "Subtle methodological variations can substantially impact correlation test performance in ecological time series"

Folder Contents
---------------

    generate_data.py: Generate simulated time series data.
    statistics.py: Functions for each correlation statistic. Includes analytical tests.
    clifford_ttest.py: Function for the parametric t-test and its helper functions.
    ccm.py: Function for cross map skill and its helper functions.
    make_surrogates.py: Functions to generate surrogate data.
    dependence_test_suite.py: Functions to run all dependence tests.

Code Dependencies
-----------------
Python 3.x (I use 3.8.5)
NumPy v1.23.0 or newer
SciPy v1.7.0 or newer (scipy.integrate.solve_ivp needs to support the "args" argument)
scikit-learn v0.18.0 or newer
pyunicorn v0.6.1 (I had trouble installing this, look at the github page for help)
pandas version 2.0.0 or newer
statsmodels 0.14.0

If anything is not working for you, feel free to email me at cannistc@uci.edu.
