#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:57:56 2022

@author: erri
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

######################################################################################
# FUNCTIONS
######################################################################################
def interpolate(func, xData, yData, ic=None, bounds=(-np.inf, np.inf)):
    # Interpolate data by fitting a given function, then returns the interpolated curve as a 1d array.
    par, covar = opt.curve_fit(func, xData, yData, p0=ic, maxfev=8000, bounds=bounds)
    if len(par) == 2:
        intCurve = func(xData, par[0], par[1])
    elif len(par) == 3:
        intCurve = func(xData, par[0], par[1], par[2])
    elif len(par) == 4:
        intCurve = func(xData, par[0], par[1], par[2], par[3])
    else:
        print("Interpolation failed. The interpolation function must have 2 or 3 parameters")
        intCurve = -1 * np.ones(len(xData))
    return par, intCurve, covar

# Scour and deposition volumes interpolation function
def func_exp(x,A,B):
    y = A*(1-np.exp(-x/B))
    return y

def func_exp2(x,A,B,C):
    y = C + A*(1-np.exp(-x/B))
    return y

# morphW interpolation function:
def func_exp3(x,A,B):
    y = ((A + (1-np.exp(-x/B)))/(A+1))*0.8
    return y
    
def func_exp4(x,A,B,C):
    y = A*C**(x/C)
    return y

def func_ln(x,A,B):
    y=A*np.ln(x/B)
    return y

###############################################################################
# SCRIPT PARAMETERS
###############################################################################
start = time.time()
N = 4 # Number of series to extract

###############################################################################
# SETUP FOLDERS
###############################################################################

w_dir = os.getcwd() # set working directory
data_folder = os.path.join(w_dir, 'output') # Set path were source data are stored
run_dir = os.path.join(w_dir, 'surveys') # Set path were surveys for each runs are stored




# EXTRACT ALL RUNS
RUNS = [] # Initialize RUS array
for RUN in sorted(os.listdir(run_dir)):
    if RUN.startswith('q'):
        RUNS = np.append(RUNS, RUN)
    
# MANUAL RUNS SELECTION
RUNS = ['q07_1', 'q10_1', 'q10_2', 'q15_1', 'q15_2', 'q20_1', 'q20_2']

# Per ogni run leggere i dati dal file di testo dei valori di scavi, depositi e active width.
# I dati vengono posti all'interno di in dizionario'
d = {} # Dictionary for runs data. Each entry is a report (output for DoD_analysis script)
int_d = {} # Dictionary for data series to be interpolated. An estraction of data from dictionary d
param_d = {} # Dictionary of all the interpolation parameters

####
# For loop over all runs
####
for run in RUNS:
    
    # Extract the number of survey files in the survey folder
    # Needed to know the dimension of the matrix
    files = []
    for f in sorted(os.listdir(os.path.join(run_dir,run))):
        path = os.path.join(run_dir,run, f)
        if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_'+run+'s'):
            files = np.append(files, f)
    file_N = files.shape[0]-1 # Number of DoD
    
    # Create the d dictionary entry loading report files from DoD_analysis.py output folder
    d["sco_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_sco_report.txt'), delimiter = ',')
    d["dep_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_dep_report.txt'), delimiter = ',')
    d["morphWact_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_morphWact_report.txt'), delimiter = ',')
    
    # Fill int_d dictionary with an extraction of scour, deposition and morphological active width reports
    # The script create a number of N indipendent sereis of data and append the last file_N - N as the averaged values.
    # For N = 4 and file_N = 9
    # DoD step0 | 1-0  | 2-1 |  3-2 |  4-3 |  5-4   6-5   7-6   8-7   9-8   average0   STDEV
    # DoD step1 | 2-0  | 3-1 |  4-2 |  5-3 |  6-4   7-5   8-6   9-7         average1   STDEV
    # DoD step2 | 3-0  | 4-1 |  5-2 |  6-3 |  7-4   8-5   9-6               average2   STDEV
    # DoD step3 | 4-0  | 5-1 |  6-2 |  7-3 |  8-4   9-5                     average3   STDEV
    # DoD step4 | 5-0  | 6-1 |  7-2 |  8-3 |  9-4                           average4   STDEV
    # DoD step5   6-0   7-1   8-2   9-3                                   | average5 | STDEV
    # DoD step6   7-0   8-1   9-2                                         | average6 | STDEV
    # DoD step7   8-0   9-1                                               | average7 | STDEV
    # DoD step8   9-0                                                     | average8 | STDEV
    # as:
    # |   1-0    |    2-1   |    3-2   |    4-3   |
    # |   2-0    |    3-1   |    4-2   |    5-3   |
    # |   3-0    |    4-1   |    5-2   |    6-3   |
    # |   4-0    |    5-1   |    6-2   |    7-3   |
    # |   5-0    |    6-1   |    7-2   |    8-3   |
    # | average5 | average5 | average5 | average5 |
    # | average6 | average6 | average6 | average6 |
    # | average7 | average7 | average7 | average7 |
    # | average8 | average8 | average8 | average8 |
    
    # Create dictionary entries with zeros
    int_d["int_sco_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_dep_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_morphWact_data_{0}".format(run)] = np.zeros((file_N, N))
    
    # Fill int_d dictionary with data as above
    for i in range(0,N):
        int_d["int_dep_data_" + run][:file_N-N+1,:N] = d["dep_data_" + run][:file_N-N+1,:N]
        int_d["int_dep_data_" + run][file_N-N+i,:] = d["dep_data_" + run][file_N-N+i,-2]
        int_d["int_sco_data_" + run][:file_N-N+1,:N] = d["sco_data_" + run][:file_N-N+1,:N]
        int_d["int_sco_data_" + run][file_N-N+i,:] = d["sco_data_" + run][file_N-N+i,-2]
        int_d["int_morphWact_data_" + run][:file_N-N+1,:N] = d["morphWact_data_" + run][:file_N-N+1,:N]
        int_d["int_morphWact_data_" + run][file_N-N+i,:] = d["morphWact_data_" + run][file_N-N+i,-2]
        
    # INTERPOLATION
    param_d["param_int_{0}".format(run)] = np.zeros((2,2)) # Initialize dictionary
    xData = int_d["int_dep_data_" + run][:,0]
    dep_par, dep_intCurve, dep_covar =  interpolate(func_exp, xData, np.linspace(0,int(file_N), 1), ic=None, bounds=(-np.inf, np.inf))
    
    
    
# Per ogni run plottare i grafici degli andamenti interpolare ogni curva con la propria funzione

# Stampare in un file txt i valori dei parametri della funzione interpolatrice e la loro deviazione standard



























end = time.time()
print()
print('Execution time: ', (end-start), 's')