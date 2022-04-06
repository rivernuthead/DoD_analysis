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

start = time.time() # Set initial time
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
# SETUP FOLDERS
###############################################################################

w_dir = os.getcwd() # set working directory
data_folder = os.path.join(w_dir, 'output') # Set path were source data are stored
run_dir = os.path.join(w_dir, 'surveys') # Set path were surveys for each runs are stored




# EXTRACT RUNS
RUNS = [] # Initialize RUS array
for RUN in sorted(os.listdir(run_dir)):
    if RUN.startswith('q') and RUN.endswith('_2'):
        RUNS = np.append(RUNS, RUN)

# Per ogni run leggere i dati dal file di testo dei valori di scavi, depositi e active width.
# I dati vengono posti all'interno di in dizionario'
d = {}
for run in RUNS:
    sco_report = np.loadtxt(os.path.join(data_folder, run + '_sco_report.txt'), delimiter = ',')
    dep_report = np.loadtxt(os.path.join(data_folder, run + '_dep_report.txt'), delimiter = ',')
    morphW_report = np.loadtxt(os.path.join(data_folder, run + '_morphWact_report.txt'), delimiter = ',')
    d["sco_data_{0}".format(run)] = sco_report[:-3,:4]
    d["dep_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_dep_report.txt'), delimiter = ',')
    d["morphWact_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_morphWact_report.txt'), delimiter = ',')
    

# Per ogni run plottare i grafici degli andamenti interpolare ogni curva con la propria funzione

# Stampare in un file txt i valori dei parametri della funzione interpolatrice e la loro deviazione standard



























end = time.time()
print()
print('Execution time: ', (end-start), 's')