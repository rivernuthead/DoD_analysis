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
    pars=[]
    if len(par) == 2:
        intCurve = func(xData, par[0], par[1])
    elif len(par) == 3:
        intCurve = func(xData, par[0], par[1], par[2])
    elif len(par) == 4:
        intCurve = func(xData, par[0], par[1], par[2], par[3])
    else:
        print("Interpolation failed. The interpolation function must have 2 or 3 parameters")
        intCurve = -1 * np.ones(len(xData))
    for i in range (0,len(par)):
        pars = np.append(pars, (par[i],covar[i,i])) 
    return par, intCurve, covar, pars

# Scour and deposition volumes interpolation function
def func_exp(x,A,B):
    # func_mode = 1
    y = A*(1-np.exp(-x/B))
    return y

def func_exp2(x,A,B,C):
    # func_mode = 2
    y = C + A*(1-np.exp(-x/B))
    return y

# morphW interpolation function:
def func_exp3(x,A,B):
    # func_mode = 3
    y = ((A + (1-np.exp(-x/B)))/(A+1))*0.7
    return y

def func_exp4(x,A,B):
    # func_mode = 4
    y = 2 + A*(1-np.exp(-x/B))
    return y

###############################################################################
# SCRIPT PARAMETERS
###############################################################################
start = time.time()
N = 4 # Number of series to extract

# Plots mode
plot_N_mode = 0 # Enable (1) or disable (0) plot print
plot_mean = 1 # Plot mean interpolation
mode_mode = 0 # what is this section????

# Interpolation function:
volume_func_mode = 1
morphWact_func_mode = 1
act_thickness_func_mode = 1
act_area_func_mode = 1

font = {'family': 'serif',
        'color':  'black',
        'weight': 'regular',
        'size': 12
        }

###############################################################################
# SETUP FOLDERS
###############################################################################

w_dir = os.getcwd() # set working directory
data_folder = os.path.join(w_dir, 'output') # Set path were source data are stored
run_dir = os.path.join(w_dir, 'surveys') # Set path were surveys for each runs are stored
plot_dir = os.path.join(w_dir, 'interpolation_plot')
int_report_dir = os.path.join(w_dir, 'int_report')
if not(os.path.exists(plot_dir)):
    os.mkdir(plot_dir)
if not(os.path.exists(int_report_dir)):
    os.mkdir(int_report_dir)

# Create separate folders for output plot
if not(os.path.exists(os.path.join(plot_dir, 'sco_interp'))):
    os.mkdir(os.path.join(plot_dir, 'sco_interp'))
if not(os.path.exists(os.path.join(plot_dir, 'dep_interp'))):
    os.mkdir(os.path.join(plot_dir, 'dep_interp'))
if not(os.path.exists(os.path.join(plot_dir, 'morphWact_interp'))):
    os.mkdir(os.path.join(plot_dir, 'morphWact_interp'))
if not(os.path.exists(os.path.join(plot_dir, 'act_thickness_interp'))):
    os.mkdir(os.path.join(plot_dir, 'act_thickness_interp'))
if not(os.path.exists(os.path.join(plot_dir, 'act_area_interp'))):
    os.mkdir(os.path.join(plot_dir, 'act_area_interp'))
if not(os.path.exists(os.path.join(plot_dir, 'vol_interp'))):
    os.mkdir(os.path.join(plot_dir, 'vol_interp'))

# EXTRACT ALL RUNS
RUNS = [] # Initialize RUS array
for RUN in sorted(os.listdir(run_dir)):
    if RUN.startswith('q'):
        RUNS = np.append(RUNS, RUN)
    
# MANUAL RUNS SELECTION
RUNS = ['q07_1', 'q10_1', 'q10_2', 'q10_3', 'q10_4', 'q15_1', 'q15_2', 'q20_1', 'q20_2']

# Per ogni run leggere i dati dal file di testo dei valori di scavi, depositi e active width e active thickness.
# I dati vengono posti all'interno di in dizionario.
d = {} # Dictionary for runs data. Each entry is a report (output for DoD_analysis script)
int_d = {} # Dictionary for data series to be interpolated. An estraction of data from dictionary d
param_d = {} # Dictionary of all the interpolation parameters

###############################################################################
# Loop over all runs
###############################################################################
for run in RUNS:
    
    # Extract the number of survey files in the survey folder
    # Needed to know the dimension of the matrix
    files = []
    for f in sorted(os.listdir(os.path.join(run_dir,run))):
        path = os.path.join(run_dir,run, f)
        if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_'+run+'s'):
            files = np.append(files, f)
    file_N = files.shape[0]-1 # Number of DoD
    
    # Load parameter matrix
    # discharge [l/s],repetition,run time [min],Texner discretization, Channel width [m], slome [m/m]
    parameters = np.loadtxt(os.path.join(w_dir, 'parameters.txt'),
                            delimiter=',',
                            skiprows=1)
    # Extract run parameter
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]

    dt = run_param[0,2] # dt between runs in minutes (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    W = run_param[0,4] # Flume width [m]
    
    
    # Create the d dictionary entry loading report files from DoD_analysis.py output folder
    d["sum_vol_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_sum_vol_report.txt'), delimiter = ',')
    d["sco_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_sco_report.txt'), delimiter = ',')
    d["dep_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_dep_report.txt'), delimiter = ',')
    d["morphWact_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_morphWact_report.txt'), delimiter = ',')
    d["act_thickness_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_act_thickness_report.txt'), delimiter = ',')
    d["act_thickness_data_dep_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_act_thickness_report_dep.txt'), delimiter = ',')
    d["act_thickness_data_sco_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_act_thickness_report_sco.txt'), delimiter = ',')
    d["act_area_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_act_area_report.txt'), delimiter = ',')
    d["act_area_data_dep_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_act_area_report_dep.txt'), delimiter = ',')
    d["act_area_data_sco_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run, run + '_act_area_report_sco.txt'), delimiter = ',')
    
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
    int_d["int_sum_vol_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_sco_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_dep_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_morphWact_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_act_thickness_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_act_thickness_data_dep_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_act_thickness_data_sco_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_act_area_data_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_act_area_data_dep_{0}".format(run)] = np.zeros((file_N, N))
    int_d["int_act_area_data_sco_{0}".format(run)] = np.zeros((file_N, N))
    
    # Fill int_d dictionary with data as above
    # When the dictionary is filled it's possible to perform interpolation
    # Serie1 Serie2 Serie3 Serie4 Mean
    for i in range(0,N):
        int_d["int_sum_vol_data_" + run][:file_N-N+1,:N] = d["sum_vol_data_" + run][:file_N-N+1,:N]
        int_d["int_sum_vol_data_" + run][file_N-N+i,:] = d["sum_vol_data_" + run][file_N-N+i,-2]
        
        int_d["int_dep_data_" + run][:file_N-N+1,:N] = d["dep_data_" + run][:file_N-N+1,:N]
        int_d["int_dep_data_" + run][file_N-N+i,:] = d["dep_data_" + run][file_N-N+i,-2]
        
        int_d["int_sco_data_" + run][:file_N-N+1,:N] = d["sco_data_" + run][:file_N-N+1,:N]
        int_d["int_sco_data_" + run][file_N-N+i,:] = d["sco_data_" + run][file_N-N+i,-2]
        
        int_d["int_morphWact_data_" + run][:file_N-N+1,:N] = d["morphWact_data_" + run][:file_N-N+1,:N]
        int_d["int_morphWact_data_" + run][file_N-N+i,:] = d["morphWact_data_" + run][file_N-N+i,-2]
        
        int_d["int_act_thickness_data_" + run][:file_N-N+1,:N] = d["act_thickness_data_" + run][:file_N-N+1,:N]
        int_d["int_act_thickness_data_" + run][file_N-N+i,:] = d["act_thickness_data_" + run][file_N-N+i,-2]
        
        int_d["int_act_thickness_data_dep_" + run][:file_N-N+1,:N] = d["act_thickness_data_dep_" + run][:file_N-N+1,:N]
        int_d["int_act_thickness_data_dep_" + run][file_N-N+i,:] = d["act_thickness_data_dep_" + run][file_N-N+i,-2]
        
        int_d["int_act_thickness_data_sco_" + run][:file_N-N+1,:N] = d["act_thickness_data_sco_" + run][:file_N-N+1,:N]
        int_d["int_act_thickness_data_sco_" + run][file_N-N+i,:] = d["act_thickness_data_sco_" + run][file_N-N+i,-2]
        
        int_d["int_act_area_data_" + run][:file_N-N+1,:N] = d["act_area_data_" + run][:file_N-N+1,:N]
        int_d["int_act_area_data_" + run][file_N-N+i,:] = d["act_area_data_" + run][file_N-N+i,-2]
        
        int_d["int_act_area_data_dep_" + run][:file_N-N+1,:N] = d["act_area_data_dep_" + run][:file_N-N+1,:N]
        int_d["int_act_area_data_dep_" + run][file_N-N+i,:] = d["act_area_data_dep_" + run][file_N-N+i,-2]
        
        int_d["int_act_area_data_sco_" + run][:file_N-N+1,:N] = d["act_area_data_sco_" + run][:file_N-N+1,:N]
        int_d["int_act_area_data_sco_" + run][file_N-N+i,:] = d["act_area_data_sco_" + run][file_N-N+i,-2]

    ###########################################################################
    #   CYCLE OVER THE NUMBER OF SERIES
    ###########################################################################
    # Build up x_data array
    xData = np.arange(1, file_N+1, 1)*dt # Time in Txnr
    
    # Perform a first interpolation on a single release as the mean of the dataset
    # TOTAL VOLUME INTERPOLATION AS SUM OF DEPOSITION AND SCOUR
    sum_vol_yData_mean = d["sum_vol_data_" + run][:,-2]
    dep_yData_mean = d["dep_data_" + run][:-4,-2]
    sco_yData_mean = np.abs(d["sco_data_" + run][:-4,-2])
    morphWact_yData_mean= d["morphWact_data_" + run][:,-2]
    act_thickness_yData_mean = d["act_thickness_data_" + run][:,-2]
    act_thickness_yData_dep_mean = d["act_thickness_data_dep_" + run][:,-2]
    act_thickness_yData_sco_mean = d["act_thickness_data_sco_" + run][:,-2]
    act_area_yData_mean = d["act_area_data_" + run][:-4,-2]
    act_area_dep_yData_mean = d["act_area_data_dep_" + run][:-4,-2]
    act_area_sco_yData_mean = d["act_area_data_sco_" + run][:-4,-2]
    
    # Interpolation process:
    # TOTAL VOLUME INTERPOLATION AS SUM OF DEPOSITION AND SCOUR
    if volume_func_mode == 1:
        sum_vol_mean_ic=np.array([np.max(sum_vol_yData_mean), np.min(xData)]) # Initial deposition parameter guess
        sum_vol_mean_par, sum_vol_mean_intCurve, sum_vol_mean_covar, sum_vol_mean_params_interp =  interpolate(func_exp, xData, sum_vol_yData_mean, ic=sum_vol_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        sum_vol_mean_ic=np.array([np.max(sum_vol_yData_mean) - np.min(sum_vol_yData_mean)/2,np.min(xData), np.min(sum_vol_yData_mean)/2]) # Initial deposition parameter guess
        sum_vol_mean_par, sum_vol_mean_intCurve, sum_vol_mean_covar, sum_vol_mean_params_interp =  interpolate(func_exp2, xData, sum_vol_yData_mean, ic=sum_vol_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        sum_vol_mean_ic=np.array([np.min(sum_vol_yData_mean),np.min(xData)]) # Initial deposition parameter guess
        sum_vol_mean_par, sum_vol_mean_intCurve, sum_vol_mean_covar, sum_vol_mean_params_interp =  interpolate(func_exp3, xData, sum_vol_yData_mean, ic=sum_vol_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     sum_vol_mean_params = sum_vol_mean_params_interp
    # else:
    #     params = sum_vol_mean_params_interp
    #     sum_vol_mean_params = np.row_stack((sum_vol_mean_params,params))
    
    # DEPOSITION VOLUME INTERPOLATION
    if volume_func_mode == 1:
        dep_mean_ic=np.array([np.max(dep_yData_mean), np.min(xData)]) # Initial deposition parameter guess
        dep_mean_par, dep_mean_intCurve, dep_mean_covar, dep_mean_params_interp =  interpolate(func_exp, xData, dep_yData_mean, ic=dep_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        dep_mean_ic=np.array([np.max(dep_yData_mean) - np.min(dep_yData_mean)/2,np.min(xData), np.min(dep_yData_mean)/2]) # Initial deposition parameter guess
        dep_mean_par, dep_mean_intCurve, dep_mean_covar, dep_mean_params_interp =  interpolate(func_exp2, xData, dep_yData_mean, ic=dep_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        dep_mean_ic=np.array([np.min(dep_yData_mean),np.min(xData)]) # Initial deposition parameter guess
        dep_mean_par, dep_mean_intCurve, dep_mean_covar, dep_mean_params_interp =  interpolate(func_exp3, xData, dep_yData_mean, ic=dep_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     dep_mean_params = dep_mean_params_interp
    # else:
    #     params = dep_mean_params_interp
    #     dep_mean_params = np.row_stack((dep_mean_params,params))
    
    # SCOUR VOLUME INTERPOLATION
    if volume_func_mode == 1:
        sco_mean_ic=np.array([np.max(sco_yData_mean), np.min(xData)]) # Initial scoosition parameter guess
        sco_mean_par, sco_mean_intCurve, sco_mean_covar, sco_mean_params_interp =  interpolate(func_exp, xData, sco_yData_mean, ic=sco_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        sco_mean_ic=np.array([np.max(sco_yData_mean) - np.min(sco_yData_mean)/2,np.min(xData), np.min(sco_yData_mean)/2]) # Initial scoosition parameter guess
        sco_mean_par, sco_mean_intCurve, sco_mean_covar, sco_mean_params_interp =  interpolate(func_exp2, xData, sco_yData_mean, ic=sco_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        sco_mean_ic=np.array([np.min(sco_yData_mean),np.min(xData)]) # Initial scoosition parameter guess
        sco_mean_par, sco_mean_intCurve, sco_mean_covar, sco_mean_params_interp =  interpolate(func_exp3, xData, sco_yData_mean, ic=sco_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     sco_mean_params = sco_mean_params_interp
    # else:
    #     params = sco_mean_params_interp
    #     sco_mean_params = np.row_stack((sco_mean_params,params))
        
    # MORPHOLOGICAL ACTIVE WIDTH INTERPOLATION
    if volume_func_mode == 1:
        morphWact_mean_ic=np.array([np.max(morphWact_yData_mean), np.min(xData)]) # Initial morphWactosition parameter guess
        morphWact_mean_par, morphWact_mean_intCurve, morphWact_mean_covar, morphWact_mean_params_interp =  interpolate(func_exp, xData, morphWact_yData_mean, ic=morphWact_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        morphWact_mean_ic=np.array([np.max(morphWact_yData_mean) - np.min(morphWact_yData_mean)/2,np.min(xData), np.min(morphWact_yData_mean)/2]) # Initial morphWactosition parameter guess
        morphWact_mean_par, morphWact_mean_intCurve, morphWact_mean_covar, morphWact_mean_params_interp =  interpolate(func_exp2, xData, morphWact_yData_mean, ic=morphWact_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        morphWact_mean_ic=np.array([np.min(morphWact_yData_mean),np.min(xData)]) # Initial morphWactosition parameter guess
        morphWact_mean_par, morphWact_mean_intCurve, morphWact_mean_covar, morphWact_mean_params_interp =  interpolate(func_exp3, xData, morphWact_yData_mean, ic=morphWact_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     morphWact_mean_params = morphWact_mean_params_interp
    # else:
    #     params = morphWact_mean_params_interp
    #     morphWact_mean_params = np.row_stack((morphWact_mean_params,params))
    
    # ACTIVE THICKNESS INTERPOLATION
    if volume_func_mode == 1:
        act_thickness_mean_ic=np.array([np.max(act_thickness_yData_mean), np.min(xData)]) # Initial act_thicknessosition parameter guess
        act_thickness_mean_par, act_thickness_mean_intCurve, act_thickness_mean_covar, act_thickness_mean_params_interp =  interpolate(func_exp, xData, act_thickness_yData_mean, ic=act_thickness_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        act_thickness_mean_ic=np.array([np.max(act_thickness_yData_mean) - np.min(act_thickness_yData_mean)/2,np.min(xData), np.min(act_thickness_yData_mean)/2]) # Initial act_thicknessosition parameter guess
        act_thickness_mean_par, act_thickness_mean_intCurve, act_thickness_mean_covar, act_thickness_mean_params_interp =  interpolate(func_exp2, xData, act_thickness_yData_mean, ic=act_thickness_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        act_thickness_mean_ic=np.array([np.min(act_thickness_yData_mean),np.min(xData)]) # Initial act_thicknessosition parameter guess
        act_thickness_mean_par, act_thickness_mean_intCurve, act_thickness_mean_covar, act_thickness_mean_params_interp =  interpolate(func_exp3, xData, act_thickness_yData_mean, ic=act_thickness_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     act_thickness_mean_params = act_thickness_mean_params_interp
    # else:
    #     params = act_thickness_mean_params_interp
    #     act_thickness_mean_params = np.row_stack((act_thickness_mean_params,params))
    
    # ACTIVE DEPOSITION THICKNESS INTERPOLATION
    if volume_func_mode == 1:
        act_thickness_dep_mean_ic=np.array([np.max(act_thickness_yData_dep_mean), np.min(xData)]) # Initial act_thicknessosition parameter guess
        act_thickness_dep_mean_par, act_thickness_dep_mean_intCurve, act_thickness_dep_mean_covar, act_thickness_dep_mean_params_interp =  interpolate(func_exp, xData, act_thickness_yData_dep_mean, ic=act_thickness_dep_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        act_thickness_dep_mean_ic=np.array([np.max(act_thickness_yData_dep_mean) - np.min(act_thickness_yData_dep_mean)/2,np.min(xData), np.min(act_thickness_yData_dep_mean)/2]) # Initial act_thicknessosition parameter guess
        act_thickness_dep_mean_par, act_thickness_dep_mean_intCurve, act_thickness_dep_mean_covar, act_thickness_dep_mean_params_interp =  interpolate(func_exp2, xData, act_thickness_yData_dep_mean, ic=act_thickness_dep_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        act_thickness_dep_mean_ic=np.array([np.min(act_thickness_yData_dep_mean),np.min(xData)]) # Initial act_thicknessosition parameter guess
        act_thickness_dep_mean_par, act_thickness_dep_mean_intCurve, act_thickness_dep_mean_covar, act_thickness_dep_mean_params_interp =  interpolate(func_exp3, xData, act_thickness_yData_dep_mean, ic=act_thickness_dep_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     act_thickness_mean_params = act_thickness_mean_params_interp
    # else:
    #     params = act_thickness_mean_params_interp
    #     act_thickness_mean_params = np.row_stack((act_thickness_mean_params,params))
        
    # ACTIVE SCOUR THICKNESS INTERPOLATION
    if volume_func_mode == 1:
        act_thickness_sco_mean_ic=np.array([np.max(act_thickness_yData_sco_mean), np.min(xData)]) # Initial act_thicknessosition parameter guess
        act_thickness_sco_mean_par, act_thickness_sco_mean_intCurve, act_thickness_sco_mean_covar, act_thickness_sco_mean_params_interp =  interpolate(func_exp, xData, act_thickness_yData_sco_mean, ic=act_thickness_sco_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        act_thickness_sco_mean_ic=np.array([np.max(act_thickness_yData_sco_mean) - np.min(act_thickness_yData_sco_mean)/2,np.min(xData), np.min(act_thickness_yData_sco_mean)/2]) # Initial act_thicknessosition parameter guess
        act_thickness_sco_mean_par, act_thickness_sco_mean_intCurve, act_thickness_sco_mean_covar, act_thickness_sco_mean_params_interp =  interpolate(func_exp2, xData, act_thickness_yData_sco_mean, ic=act_thickness_sco_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        act_thickness_sco_mean_ic=np.array([np.min(act_thickness_yData_sco_mean),np.min(xData)]) # Initial act_thicknessosition parameter guess
        act_thickness_sco_mean_par, act_thickness_sco_mean_intCurve, act_thickness_sco_mean_covar, act_thickness_sco_mean_params_interp =  interpolate(func_exp3, xData, act_thickness_yData_sco_mean, ic=act_thickness_sco_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     act_thickness_mean_params = act_thickness_mean_params_interp
    # else:
    #     params = act_thickness_mean_params_interp
    #     act_thickness_mean_params = np.row_stack((act_thickness_mean_params,params))
        
    # ACTIVE AREA INTERPOLATION
    if volume_func_mode == 1:
        act_area_mean_ic=np.array([np.max(act_area_yData_mean), np.min(xData)]) # Initial act_areaosition parameter guess
        act_area_mean_par, act_area_mean_intCurve, act_area_mean_covar, act_area_mean_params_interp =  interpolate(func_exp, xData, act_area_yData_mean, ic=act_area_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        act_area_mean_ic=np.array([np.max(act_area_yData_mean) - np.min(act_area_yData_mean)/2,np.min(xData), np.min(act_area_yData_mean)/2]) # Initial act_areaosition parameter guess
        act_area_mean_par, act_area_mean_intCurve, act_area_mean_covar, act_area_mean_params_interp =  interpolate(func_exp2, xData, act_area_yData_mean, ic=act_area_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        act_area_mean_ic=np.array([np.min(act_area_yData_mean),np.min(xData)]) # Initial act_areaosition parameter guess
        act_area_mean_par, act_area_mean_intCurve, act_area_mean_covar, act_area_mean_params_interp =  interpolate(func_exp3, xData, act_area_yData_mean, ic=act_area_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     act_area_mean_params = act_area_mean_params_interp
    # else:
    #     params = act_area_mean_params_interp
    #     act_area_mean_params = np.row_stack((act_area_mean_params,params))
        
    # ACTIVE DEPOSITION AREA INTERPOLATION
    if volume_func_mode == 1:
        act_area_dep_mean_ic=np.array([np.max(act_area_dep_yData_mean), np.min(xData)]) # Initial act_area_deposition parameter guess
        act_area_dep_mean_par, act_area_dep_mean_intCurve, act_area_dep_mean_covar, act_area_dep_mean_params_interp =  interpolate(func_exp, xData, act_area_dep_yData_mean, ic=act_area_dep_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        act_area_dep_mean_ic=np.array([np.max(act_area_dep_yData_mean) - np.min(act_area_dep_yData_mean)/2,np.min(xData), np.min(act_area_dep_yData_mean)/2]) # Initial act_area_deposition parameter guess
        act_area_dep_mean_par, act_area_dep_mean_intCurve, act_area_dep_mean_covar, act_area_dep_mean_params_interp =  interpolate(func_exp2, xData, act_area_dep_yData_mean, ic=act_area_dep_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        act_area_dep_mean_ic=np.array([np.min(act_area_dep_yData_mean),np.min(xData)]) # Initial act_area_deposition parameter guess
        act_area_dep_mean_par, act_area_dep_mean_intCurve, act_area_dep_mean_covar, act_area_dep_mean_params_interp =  interpolate(func_exp3, xData, act_area_dep_yData_mean, ic=act_area_dep_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     act_area_dep_mean_params = act_area_dep_mean_params_interp
    # else:
    #     params = act_area_dep_mean_params_interp
    #     act_area_dep_mean_params = np.row_stack((act_area_dep_mean_params,params))
        
    # ACTIVE SCOUR AREA INTERPOLATION
    if volume_func_mode == 1:
        act_area_sco_mean_ic=np.array([np.max(act_area_sco_yData_mean), np.min(xData)]) # Initial act_area_scoosition parameter guess
        act_area_sco_mean_par, act_area_sco_mean_intCurve, act_area_sco_mean_covar, act_area_sco_mean_params_interp =  interpolate(func_exp, xData, act_area_sco_yData_mean, ic=act_area_sco_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode ==2:
        act_area_sco_mean_ic=np.array([np.max(act_area_sco_yData_mean) - np.min(act_area_sco_yData_mean)/2,np.min(xData), np.min(act_area_sco_yData_mean)/2]) # Initial act_area_scoosition parameter guess
        act_area_sco_mean_par, act_area_sco_mean_intCurve, act_area_sco_mean_covar, act_area_sco_mean_params_interp =  interpolate(func_exp2, xData, act_area_sco_yData_mean, ic=act_area_sco_mean_ic, bounds=(-np.inf, np.inf))
    elif volume_func_mode == 3:
        act_area_sco_mean_ic=np.array([np.min(act_area_sco_yData_mean),np.min(xData)]) # Initial act_area_scoosition parameter guess
        act_area_sco_mean_par, act_area_sco_mean_intCurve, act_area_sco_mean_covar, act_area_sco_mean_params_interp =  interpolate(func_exp3, xData, act_area_sco_yData_mean, ic=act_area_sco_mean_ic, bounds=(-np.inf, np.inf))
    
    # if i == 0:
    #     act_area_sco_mean_params = act_area_sco_mean_params_interp
    # else:
    #     params = act_area_sco_mean_params_interp
    #     act_area_sco_mean_params = np.row_stack((act_area_sco_mean_params,params))
    
    # Plot
    if plot_mean == 1:
        
        # Deposition volume plot - MEAN
        # TODO
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, dep_yData_mean, 'o', c='blue')
        axs.plot(xData, dep_mean_intCurve, c='green')
        axs.set_title('Deposition mean interpolation ' +run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Volume V/(L*W) [mm]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(dep_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(dep_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(dep_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'dep_interp', run + '_func_mode' + str(volume_func_mode) + 'mean_dep_interp.pdf'), dpi=200)
        plt.show()
        
        # Scour volume plot - MEAN
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, sco_yData_mean, 'o', c='blue')
        axs.plot(xData, sco_mean_intCurve, c='green')
        axs.set_title('Scour mean interpolation ' +run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Volume V/(L*W) [mm]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(sco_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(sco_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(sco_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'sco_interp', run + '_func_mode' + str(volume_func_mode) + 'mean_sco_interp.pdf'), dpi=200)
        plt.show()
        
        # Morphological active width plot - MEAN
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, morphWact_yData_mean, 'o', c='blue')
        axs.plot(xData, morphWact_mean_intCurve, c='green')
        axs.set_title('morphWact mean interpolation ' +run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Morphological active width [%]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(morphWact_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(morphWact_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(morphWact_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'morphWact_interp', run + '_func_mode' + str(volume_func_mode) + 'mean_morphWact_interp.pdf'), dpi=200)
        plt.show()
        
        # Active thickness plot - MAIN
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, act_thickness_yData_mean, 'o', c='blue')
        axs.plot(xData, act_thickness_mean_intCurve, c='green')
        axs.set_title('act_thickness mean interpolation ' +run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Morphological active layer [mm]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(act_thickness_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(act_thickness_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_thickness_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'act_thickness_interp', run + '_func_mode' + str(volume_func_mode) + 'mean_act_thickness_interp.pdf'), dpi=200)
        plt.show()
        
        # Active deposition thickness plot - MEAN
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, act_thickness_yData_dep_mean, 'o', c='blue')
        axs.plot(xData, act_thickness_dep_mean_intCurve, c='green')
        axs.set_title('act_thickness dep mean interpolation ' +run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Morphological active layer [mm]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(act_thickness_dep_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(act_thickness_dep_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_thickness_dep_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'act_thickness_interp', run + '_func_mode' + str(volume_func_mode) + 'dep_mean_act_thickness_interp.pdf'), dpi=200)
        plt.show()
        
        # Active scour thickness plot - MEAN
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, act_thickness_yData_sco_mean, 'o', c='blue')
        axs.plot(xData, act_thickness_sco_mean_intCurve, c='green')
        axs.set_title('act_thickness sco mean interpolation ' +run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Morphological active layer [mm]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(act_thickness_sco_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(act_thickness_sco_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_thickness_sco_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'act_thickness_interp', run + '_func_mode' + str(volume_func_mode) + 'sco_mean_act_thickness_interp.pdf'), dpi=200)
        plt.show()
        
        # Active area interpolation - MEAN
        fig7, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
        axs.plot(xData, act_area_yData_mean, 'o', c='pink')
        axs.plot(xData, act_area_mean_intCurve, c='green')
        axs.set_title('Active area mean interpolation '+run)
        axs.set_xlabel('Time [min]')
        axs.set_ylabel('Active area [mm²]')
        axs.set_ylim(bottom=0)
        # axs.set_xlim(bottom=0)
        # plt.text(np.max(xData)*0.7, np.min(act_area_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$='    + str(np.round(act_area_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_area_mean_params_interp[0], decimals=1)), fontdict=font)
        plt.savefig(os.path.join(plot_dir, 'act_area_interp', run + '_func_mode' + str(act_area_func_mode) + 'series_' + 'mean_act_area_interp.pdf'), dpi=200)
        plt.show()
        
    if mode_mode==1:    
        for i in range(0,N):
            
            xData = np.arange(1, file_N+1, 1)*dt # Time in Txnr
            #TODO Check parameter boundaries https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
            
            
            # TOTAL VOLUME INTERPOLATION AS SUM OF DEPOSITION AND SCOUR
            sum_vol_yData = int_d["int_sum_vol_data_" + run][:,i]
            # Decide interpolation function
            if volume_func_mode == 1:
                sum_vol_ic=np.array([np.max(sum_vol_yData), np.min(xData)]) # Initial deposition parameter guess
                sum_vol_par, sum_vol_intCurve, sum_vol_covar, sum_vol_params_interp =  interpolate(func_exp, xData, sum_vol_yData, ic=sum_vol_ic, bounds=(-np.inf, np.inf))
            elif volume_func_mode ==2:
                sum_vol_ic=np.array([np.max(sum_vol_yData) - np.min(sum_vol_yData)/2,np.min(xData), np.min(sum_vol_yData)/2]) # Initial deposition parameter guess
                sum_vol_par, sum_vol_intCurve, sum_vol_covar, sum_vol_params_interp =  interpolate(func_exp2, xData, sum_vol_yData, ic=sum_vol_ic, bounds=(-np.inf, np.inf))
            elif volume_func_mode == 3:
                sum_vol_ic=np.array([np.min(sum_vol_yData),np.min(xData)]) # Initial deposition parameter guess
                sum_vol_par, sum_vol_intCurve, sum_vol_covar, sum_vol_params_interp =  interpolate(func_exp3, xData, sum_vol_yData, ic=sum_vol_ic, bounds=(-np.inf, np.inf))
                
            if i == 0:
                sum_vol_params = sum_vol_params_interp
            else:
                params = sum_vol_params_interp
                sum_vol_params = np.row_stack((sum_vol_params,params))
            
            # DEPOSITION VOLUME INTERPOLATION
            dep_yData = int_d["int_dep_data_" + run][:,i]
            # Decide interpolation function
            if volume_func_mode == 1:
                dep_ic=np.array([np.max(dep_yData), np.min(xData)]) # Initial deposition parameter guess
                dep_par, dep_intCurve, dep_covar, dep_params_interp =  interpolate(func_exp, xData, dep_yData, ic=dep_ic, bounds=(-np.inf, np.inf))
            elif volume_func_mode ==2:
                dep_ic=np.array([np.max(dep_yData) - np.min(dep_yData)/2,np.min(xData), np.min(dep_yData)/2]) # Initial deposition parameter guess
                dep_par, dep_intCurve, dep_covar, dep_params_interp =  interpolate(func_exp2, xData, dep_yData, ic=dep_ic, bounds=(-np.inf, np.inf))
            elif volume_func_mode == 3:
                dep_ic=np.array([np.min(dep_yData),np.min(xData)]) # Initial deposition parameter guess
                dep_par, dep_intCurve, dep_covar, dep_params_interp =  interpolate(func_exp3, xData, dep_yData, ic=dep_ic, bounds=(-np.inf, np.inf))
                
            if i == 0:
                dep_params = dep_params_interp
            else:
                params = dep_params_interp
                dep_params = np.row_stack((dep_params,params))
        
            
            # SCOUR VOLUME INTERPOLATION
            sco_yData = np.abs(int_d["int_sco_data_" + run][:,i])
            if volume_func_mode == 1:
                sco_ic=np.array([np.max(sco_yData),np.min(xData)]) # Initial deposition parameter guess
                sco_par, sco_intCurve, sco_covar, sco_params_interp =  interpolate(func_exp, xData, sco_yData, ic=sco_ic, bounds=(-np.inf, np.inf))
            elif volume_func_mode ==2:
                sco_ic=np.array([np.max(sco_yData) - np.min(sco_yData)/2,np.min(xData), np.min(sco_yData)/2]) # Initial deposition parameter guess
                sco_par, sco_intCurve, sco_covar, sco_params_interp =  interpolate(func_exp2, xData, sco_yData, ic=sco_ic, bounds=(-np.inf, np.inf))
            elif volume_func_mode == 3:
                sco_ic=np.array([np.min(sco_yData),np.min(xData)]) # Initial deposition parameter guess
                sco_par, sco_intCurve, sco_covar, sco_params_interp =  interpolate(func_exp3, xData, sco_yData, ic=sco_ic, bounds=(-np.inf, np.inf))
                
            if i == 0:
                sco_params = sco_params_interp
            else:
                params = sco_params_interp
                sco_params  = np.row_stack((sco_params,params))
            
                
            # MORPHOLOGICAL ACTIVE WIDTH INTERPOLATION
            morphWact_yData = int_d["int_morphWact_data_" + run][:,i]
            if morphWact_func_mode == 1:
                morphWact_ic=np.array([np.max(morphWact_yData),np.min(xData)]) # Initial active width parameter guess
                morphWact_par, morphWact_intCurve, morphWact_covar, morphWact_params_interp =  interpolate(func_exp, xData, morphWact_yData, ic=morphWact_ic, bounds=(np.array((0,-np.inf)), np.array((1,np.inf))))
            elif morphWact_func_mode ==2:
                morphWact_ic=np.array([np.max(morphWact_yData)-np.min(morphWact_yData)/2,np.min(xData), np.min(morphWact_yData)/2]) # Initial deposition parameter guess
                morphWact_par, morphWact_intCurve, morphWact_covar, morphWact_params_interp =  interpolate(func_exp2, xData, morphWact_yData, ic=morphWact_ic, bounds=(-np.inf, np.inf))
            elif morphWact_func_mode == 3:
                morphWact_ic=np.array([np.min(morphWact_yData),np.min(xData)]) # Initial active width parameter guess
                morphWact_par, morphWact_intCurve, morphWact_covar, morphWact_params_interp =  interpolate(func_exp3, xData, morphWact_yData, ic=morphWact_ic, bounds=(-np.inf, np.inf))
                    
            if i == 0:
                morphWact_params = morphWact_params_interp
            else:
                params = morphWact_params_interp
                morphWact_params = np.row_stack((morphWact_params,params))
                
            
            # ACTIVE THICKNESS INTERPOLATION
            act_thickness_yData = int_d["int_act_thickness_data_" + run][:,i]
            if act_thickness_func_mode == 1:
                act_thickness_ic=np.array([np.max(act_thickness_yData),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode ==2:
                act_thickness_ic=np.array([np.max(act_thickness_yData)-np.min(act_thickness_yData)/2,np.min(xData), np.min(act_thickness_yData)/2]) # Initial deposition parameter guess
                act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp2, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode == 3:
                act_thickness_ic=np.array([np.min(act_thickness_yData),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp3, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode == 4:
                act_thickness_ic=np.array([np.max(act_thickness_yData),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp4, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
                    
            if i == 0:
                act_thickness_params = act_thickness_params_interp
            else:
                params = act_thickness_params_interp
                act_thickness_params = np.row_stack((act_thickness_params,params))
            
            # ACTIVE DEPOSITION THICKNESS INTERPOLATION
            act_thickness_yData_dep = int_d["int_act_thickness_data_dep_" + run][:,i]
            if act_thickness_func_mode == 1:
                act_thickness_ic_dep=np.array([np.max(act_thickness_yData_dep),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par_dep, act_thickness_intCurve_dep, act_thickness_covar_dep, act_thickness_params_interp_dep =  interpolate(func_exp, xData, act_thickness_yData_dep, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode ==2:
                act_thickness_ic_dep=np.array([np.max(act_thickness_yData_dep)-np.min(act_thickness_yData_dep)/2,np.min(xData), np.min(act_thickness_yData_dep)/2]) # Initial deposition parameter guess
                act_thickness_par_dep, act_thickness_intCurve_dep, act_thickness_covar_dep, act_thickness_params_interp_dep =  interpolate(func_exp2, xData, act_thickness_yData_dep, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode == 3:
                act_thickness_ic_dep=np.array([np.min(act_thickness_yData_dep),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par_dep, act_thickness_intCurve_dep, act_thickness_covar_dep, act_thickness_params_interp_dep =  interpolate(func_exp3, xData, act_thickness_yData_dep, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode == 4:
                act_thickness_ic_dep=np.array([np.max(act_thickness_yData_dep),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par_dep, act_thickness_intCurve_dep, act_thickness_covar_dep, act_thickness_params_interp_dep =  interpolate(func_exp4, xData, act_thickness_yData_dep, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
                    
            if i == 0:
                act_thickness_params_dep = act_thickness_params_interp_dep
            else:
                params = act_thickness_params_interp_dep
                act_thickness_params_dep = np.row_stack((act_thickness_params_dep,params))
            
            
            # ACTIVE SCOUR THICKNESS INTERPOLATION
            act_thickness_yData_sco = int_d["int_act_thickness_data_sco_" + run][:,i]
            if act_thickness_func_mode == 1:
                act_thickness_ic_sco=np.array([np.max(act_thickness_yData_sco),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par_sco, act_thickness_intCurve_sco, act_thickness_covar_sco, act_thickness_params_interp_sco =  interpolate(func_exp, xData, act_thickness_yData_sco, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode ==2:
                act_thickness_ic_sco=np.array([np.max(act_thickness_yData_sco)-np.min(act_thickness_yData_sco)/2,np.min(xData), np.min(act_thickness_yData_sco)/2]) # Initial deposition parameter guess
                act_thickness_par_sco, act_thickness_intCurve_sco, act_thickness_covar_sco, act_thickness_params_interp_sco =  interpolate(func_exp2, xData, act_thickness_yData_sco, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode == 3:
                act_thickness_ic_sco=np.array([np.min(act_thickness_yData_sco),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par_sco, act_thickness_intCurve_sco, act_thickness_covar_sco, act_thickness_params_interp_sco =  interpolate(func_exp3, xData, act_thickness_yData_sco, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
            elif act_thickness_func_mode == 4:
                act_thickness_ic_sco=np.array([np.max(act_thickness_yData_sco),np.min(xData)]) # Initial deposition parameter guess
                act_thickness_par_sco, act_thickness_intCurve_sco, act_thickness_covar_sco, act_thickness_params_interp_sco =  interpolate(func_exp4, xData, act_thickness_yData_sco, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
                    
            if i == 0:
                act_thickness_params_sco = act_thickness_params_interp_sco
            else:
                params = act_thickness_params_interp_sco
                act_thickness_params_sco = np.row_stack((act_thickness_params_sco,params))
    
    
            # ACTIVE AREA INTERPOLATION
            act_area_yData = int_d["int_act_area_data_" + run][:,i]
            # Decide interpolation function
            if act_area_func_mode == 1:
                act_area_ic=np.array([np.max(act_area_yData),np.min(xData)]) # Initial act_area parameter guess
                act_area_par, act_area_intCurve, act_area_covar, act_area_params_interp =  interpolate(func_exp, xData, act_area_yData, ic=act_area_ic, bounds=(-np.inf, np.inf))
            elif act_area_func_mode ==2:
                act_area_ic=np.array([np.max(act_area_yData)-np.min(act_area_yData)/2,np.min(xData), np.min(act_area_yData)/2]) # Initial act_area parameter guess
                act_area_par, act_area_intCurve, act_area_covar, act_area_params_interp =  interpolate(func_exp2, xData, act_area_yData, ic=act_area_ic, bounds=(-np.inf, np.inf))
            elif act_area_func_mode == 3:
                act_area_ic=np.array([np.min(act_area_yData),np.min(xData)]) # Initial act_areaosition parameter guess
                act_area_par, act_area_intCurve, act_area_covar, act_area_params_interp =  interpolate(func_exp3, xData, act_area_yData, ic=act_area_ic, bounds=(-np.inf, np.inf))
                
            if i == 0:
                act_area_params = act_area_params_interp
            else:
                params = act_area_params_interp
                act_area_params = np.row_stack((act_area_params,params))
                
            # ACTIVE DEPOSITION AREA INTERPOLATION
            act_area_dep_yData = int_d["int_act_area_data_dep_" + run][:,i]
            # Decide interpolation function
            if act_area_func_mode == 1:
                act_area_dep_ic=np.array([np.max(act_area_dep_yData),np.min(xData)]) # Initial act_area parameter guess
                act_area_dep_par, act_area_dep_intCurve, act_area_dep_covar, act_area_dep_params_interp =  interpolate(func_exp, xData, act_area_dep_yData, ic=act_area_dep_ic, bounds=(-np.inf, np.inf))
            elif act_area_func_mode ==2:
                act_area_dep_ic=np.array([np.max(act_area_dep_yData)-np.min(act_area_dep_yData)/2,np.min(xData), np.min(act_area_dep_yData)/2]) # Initial act_area parameter guess
                act_area_dep_par, act_area_dep_intCurve, act_area_dep_covar, act_area_dep_params_interp =  interpolate(func_exp2, xData, act_area_dep_yData, ic=act_area_dep_ic, bounds=(-np.inf, np.inf))
            elif act_area_func_mode == 3:
                act_area_dep_ic=np.array([np.min(act_area_dep_yData),np.min(xData)]) # Initial act_areaosition parameter guess
                act_area_dep_par, act_area_dep_intCurve, act_area_dep_covar, act_area_dep_params_interp =  interpolate(func_exp3, xData, act_area_dep_yData, ic=act_area_dep_ic, bounds=(-np.inf, np.inf))
                
            if i == 0:
                act_area_dep_params = act_area_dep_params_interp
            else:
                params = act_area_dep_params_interp
                act_area_dep_params = np.row_stack((act_area_dep_params,params))
            
            
            # ACTIVE SCOUR AREA INTERPOLATION
            act_area_sco_yData = int_d["int_act_area_data_sco_" + run][:,i]
            # Decide interpolation function
            if act_area_func_mode == 1:
                act_area_sco_ic=np.array([np.max(act_area_sco_yData),np.min(xData)]) # Initial act_area parameter guess
                act_area_sco_par, act_area_sco_intCurve, act_area_sco_covar, act_area_sco_params_interp =  interpolate(func_exp, xData, act_area_sco_yData, ic=act_area_sco_ic, bounds=(-np.inf, np.inf))
            elif act_area_func_mode ==2:
                act_area_sco_ic=np.array([np.max(act_area_sco_yData)-np.min(act_area_sco_yData)/2,np.min(xData), np.min(act_area_sco_yData)/2]) # Initial act_area parameter guess
                act_area_sco_par, act_area_sco_intCurve, act_area_sco_covar, act_area_sco_params_interp =  interpolate(func_exp2, xData, act_area_sco_yData, ic=act_area_sco_ic, bounds=(-np.inf, np.inf))
            elif act_area_func_mode == 3:
                act_area_sco_ic=np.array([np.min(act_area_sco_yData),np.min(xData)]) # Initial act_areaosition parameter guess
                act_area_sco_par, act_area_sco_intCurve, act_area_sco_covar, act_area_sco_params_interp =  interpolate(func_exp3, xData, act_area_sco_yData, ic=act_area_sco_ic, bounds=(-np.inf, np.inf))
                
            if i == 0:
                act_area_sco_params = act_area_sco_params_interp
            else:
                params = act_area_sco_params_interp
                act_area_sco_params = np.row_stack((act_area_sco_params,params))
    
    
    
            #######################################################################
            #   PLOTS
            #######################################################################
            
            
            
            if plot_N_mode == 1:
                # Deposition volume plot
                fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, dep_yData, 'o', c='blue')
                axs.plot(xData, dep_intCurve, c='green')
                axs.set_title('Deposition series # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('Volume V/(L*W) [mm]')
                plt.text(np.max(xData)*0.7, np.min(dep_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(dep_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(dep_params_interp[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'dep_interp', run + '_func_mode' + str(volume_func_mode) + 'series_' + str(i+1) +'_dep_interp.png'), dpi=200)
                plt.show()
                
                
                # Scour volume plot
                fig2, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, sco_yData, 'o', c='red')
                axs.plot(xData, sco_intCurve, c='green')
                axs.set_title('Scour series # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('Volume V/(L*W) [mm]')
                plt.text(np.max(xData)*0.7, np.min(sco_intCurve), 'Trun=' + str(dt) + 'min \n'+ r'$\tau$=' + str(np.round(sco_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(sco_params_interp[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'sco_interp', run + '_func_mode' + str(volume_func_mode) + 'series_' + str(i+1) +'_sco_interp.png'), dpi=200)
                plt.show()
                
                
                # Morphological active width plot
                fig3, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, morphWact_yData, 'o', c='brown')
                axs.plot(xData, morphWact_intCurve, c='green')
                axs.set_title('Morphological active width series # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('morphWact/W [-]')
                plt.text(np.max(xData)*0.7, np.min(morphWact_intCurve), 'Trun=' + str(dt) + 'min \n'+ r'$\tau$=' + str(np.round(morphWact_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(morphWact_params_interp[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'morphWact_interp', run + '_func_mode' + str(morphWact_func_mode) + 'series_' + str(i+1) +'_morphWact_interp.png'), dpi=200)
                plt.show()
                
                # Active thickness plot
                fig4, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, act_thickness_yData, 'o', c='purple')
                axs.plot(xData, act_thickness_intCurve, c='green')
                axs.set_title('Active thickness # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('Active thickness [mm]')
                plt.text(np.max(xData)*0.7, np.min(act_thickness_intCurve),'Trun=' + str(dt) + 'min \n'+ r'$\tau$=' + str(np.round(act_thickness_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_thickness_params_interp[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'act_thickness_interp', run + '_func_mode' + str(act_thickness_func_mode) + 'series_' + str(i+1) +'_act_thickness_interp.png'), dpi=200)
                plt.show()
                
                # Active deposition thickness plot
                fig5, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, act_thickness_yData_dep, 'o', c='purple')
                axs.plot(xData, act_thickness_intCurve_dep, c='green')
                axs.set_title('Active deposition thickness # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('Active thickness [mm]')
                plt.text(np.max(xData)*0.7, np.min(act_thickness_intCurve_dep), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(act_thickness_params_interp_dep[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_thickness_params_interp_dep[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'act_thickness_interp', run + '_func_mode' + str(act_thickness_func_mode) + 'series_' + str(i+1) +'_act_thickness_interp_dep.png'), dpi=200)
                plt.show()
                
                # Active scour thickness plot
                fig6, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, act_thickness_yData_sco, 'o', c='purple')
                axs.plot(xData, act_thickness_intCurve_sco, c='green')
                axs.set_title('Active scour thickness # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('Active thickness [mm]')
                plt.text(np.max(xData)*0.7, np.min(act_thickness_intCurve_sco), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(act_thickness_params_interp_sco[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_thickness_params_interp_sco[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'act_thickness_interp', run + '_func_mode' + str(act_thickness_func_mode) + 'series_' + str(i+1) +'_act_thickness_interp_sco.png'), dpi=200)
                plt.show()
                
                # Active area interpolation 
                fig7, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
                axs.plot(xData, act_area_yData, 'o', c='pink')
                axs.plot(xData, act_area_intCurve, c='green')
                axs.set_title('Active area # '+str(i+1)+'- '+run)
                axs.set_xlabel('Time [min]')
                axs.set_ylabel('Active area [mm²]')
                plt.text(np.max(xData)*0.7, np.min(act_area_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$='    + str(np.round(act_area_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(act_area_params_interp[0], decimals=1)), fontdict=font)
                plt.savefig(os.path.join(plot_dir, 'act_area_interp', run + '_func_mode' + str(act_area_func_mode) + 'series_' + str(i+1) +'_act_area_interp.png'), dpi=200)
                plt.show()
                
                # #  Volume interpolation as act_thickness*act_area
                # fig8, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
                # axs.plot(xData, (sum_vol_yData)*(W*14*1000), 'o', c='blue')
                # axs.plot(xData, (sum_vol_intCurve)*(W*14*1000), c='green')
                # axs.plot(xData, act_area_intCurve*act_thickness_intCurve, c='black')
                # # axs.plot(xData, act_area_sco_intCurve*act_thickness_intCurve_sco + act_area_dep_intCurve*act_thickness_intCurve_dep, c='turquoise')
                # axs.set_title('Volume interpolation as act_thickness*act_area series # '+str(i+1)+'- '+run)
                # axs.set_xlabel('Time [min]')
                # axs.set_ylabel('Volume [mm³]')
                # plt.savefig(os.path.join(plot_dir, 'vol_interp', run + 'series_' + str(i+1) +'_volume_interp.png'), dpi=200)
                # plt.text(np.max(xData)*0.7, np.min(sum_vol_intCurve*(W*14*1000)), r'$\tau$=' + str(np.round(sum_vol_params_interp[2], decimals=1)) + 'min\n' + 'Txnr=' + str(dt) + 'min', fontdict=font, bbox={
                #     #'facecolor': 'green',
                #     'alpha': 0.05, 'pad': 4.0})
                # plt.show()
    
            else:
                pass

    # Transpose arrays to obtain structure as below:
    #        Serie1    Serie2    Serie3    Serie4
    #   A    
    # SD(A)
    #   B
    # SD(B)
    #   C
    # SD(C)
    
        dep_params = np.transpose(dep_params)
        sco_params = np.transpose(sco_params)
        morphWact_params = np.transpose(morphWact_params)
        act_thickness_params = np.transpose(act_thickness_params)
        act_area_params = np.transpose(act_area_params)
        
        # Print txt reports:
        header = 'Serie1, Serie2, Serie3, Serie4' # Report header
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_dep_int_param.txt'), dep_params, delimiter=',', header = header)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_sco_int_param.txt'), sco_params, delimiter=',', header = header)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(morphWact_func_mode) + '_morphWact_int_param.txt'), morphWact_params, delimiter=',', header = header)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_thickness_func_mode) + '_act_thickness_int_param.txt'), act_thickness_params, delimiter=',', header = header)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_area_func_mode) + '_act_area_int_param.txt'), act_area_params, delimiter=',', header = header)
        
        # Print txt reports for interpolation mean:
        header_mean = 'Interpolation parameters of the mean value of each run'
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_dep_int_param_mean.txt'), dep_mean_params_interp, delimiter=',', header = header_mean)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_sco_int_param_mean.txt'), sco_mean_params_interp, delimiter=',', header = header_mean)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(morphWact_func_mode) + '_morphWact_int_param_mean.txt'), morphWact_mean_params_interp, delimiter=',', header = header_mean)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_thickness_func_mode) + '_act_thickness_int_param_mean.txt'), act_thickness_mean_params_interp, delimiter=',', header = header_mean)
        np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_area_func_mode) + '_act_area_int_param_mean.txt'), act_area_mean_params_interp, delimiter=',', header = header_mean)
        
    
end = time.time()
print()
print('Execution time: ', (end-start), 's')