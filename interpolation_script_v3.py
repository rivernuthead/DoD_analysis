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
    y = 0.2 + A*(1-np.exp(-x/B))
    return y

# morphW interpolation function:
def func_exp3(x,A,B):
    # func_mode = 3
    y = ((A + (1-np.exp(-x/B)))/(A+1))*0.8
    return y

###############################################################################
# SCRIPT PARAMETERS
###############################################################################
start = time.time()
N = 4 # Number of series to extract


plot_mode = 1 # Plot mode: 0 -> no plot, 1 -> plot

# Interpolation function:
volume_func_mode = 1
morphWact_func_mode = 1
act_thickness_func_mode = 2
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
    
    # Load parameter matrix
    # discharge [l/s],repetition,run time [min],Texner discretization, Channel width [m], slome [m/m]
    parameters = np.loadtxt(os.path.join(w_dir, 'parameters.txt'),
                            delimiter=',',
                            skiprows=1)
    # Extract run parameter
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]

    dt = run_param[0,2] # dt between runs in minutes (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    
    # Create the d dictionary entry loading report files from DoD_analysis.py output folder
    d["sco_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_sco_report.txt'), delimiter = ',')
    d["dep_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_dep_report.txt'), delimiter = ',')
    d["morphWact_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_morphWact_report.txt'), delimiter = ',')
    d["act_thickness_data_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, run + '_act_thickness_report.txt'), delimiter = ',')
    
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
    int_d["int_act_thickness_data_{0}".format(run)] = np.zeros((file_N, N))
    
    # Fill int_d dictionary with data as above
    for i in range(0,N):
        int_d["int_dep_data_" + run][:file_N-N+1,:N] = d["dep_data_" + run][:file_N-N+1,:N]
        int_d["int_dep_data_" + run][file_N-N+i,:] = d["dep_data_" + run][file_N-N+i,-2]
        int_d["int_sco_data_" + run][:file_N-N+1,:N] = d["sco_data_" + run][:file_N-N+1,:N]
        int_d["int_sco_data_" + run][file_N-N+i,:] = d["sco_data_" + run][file_N-N+i,-2]
        int_d["int_morphWact_data_" + run][:file_N-N+1,:N] = d["morphWact_data_" + run][:file_N-N+1,:N]
        int_d["int_morphWact_data_" + run][file_N-N+i,:] = d["morphWact_data_" + run][file_N-N+i,-2]
        int_d["int_act_thickness_data_" + run][:file_N-N+1,:N] = d["act_thickness_data_" + run][:file_N-N+1,:N]
        int_d["int_act_thickness_data_" + run][file_N-N+i,:] = d["act_thickness_data_" + run][file_N-N+i,-2]
        
    # INTERPOLATION
    # Initialize parameter arrays
    dep_int_param = []
    sco_int_param = []
    morphWact_int_param = []
    act_thickness_int_param = []
    
    for i in range(0,N):
        
        xData = np.arange(1, file_N+1, 1)*dt # Time in Txnr
        
        # Deposition volumes interpolation:
        dep_yData = int_d["int_dep_data_" + run][:,i]
        # Decide interpolation function
        if volume_func_mode == 1:
            dep_ic=np.array([np.mean(dep_yData),np.min(xData)]) # Initial deposition parameter guess
            dep_par, dep_intCurve, dep_covar, dep_params_interp =  interpolate(func_exp, xData, dep_yData, ic=dep_ic, bounds=(-np.inf, np.inf))
        elif volume_func_mode ==2:
            dep_ic=np.array([np.mean(dep_yData),np.min(xData), np.max(dep_yData)]) # Initial deposition parameter guess
            dep_par, dep_intCurve, dep_covar, dep_params_interp =  interpolate(func_exp2, xData, dep_yData, ic=dep_ic, bounds=(-np.inf, np.inf))
        elif volume_func_mode == 3:
            #TODO check initial guess for this function
            dep_ic=np.array([np.mean(dep_yData),np.min(xData), np.max(dep_yData)]) # Initial deposition parameter guess
            dep_par, dep_intCurve, dep_covar, dep_params_interp =  interpolate(func_exp3, xData, dep_yData, ic=dep_ic, bounds=(-np.inf, np.inf))
            
        if i == 0:
            dep_params = dep_params_interp
        else:
            params = dep_params_interp
            dep_params = np.row_stack((dep_params,params))
    
        
        # Scour volumes interpolation:
        sco_yData = np.abs(int_d["int_sco_data_" + run][:,i])
        if volume_func_mode == 1:
            sco_ic=np.array([np.mean(dep_yData),np.min(xData)]) # Initial deposition parameter guess
            sco_par, sco_intCurve, sco_covar, sco_params_interp =  interpolate(func_exp, xData, sco_yData, ic=sco_ic, bounds=(-np.inf, np.inf))
        elif volume_func_mode ==2:
            sco_ic=np.array([np.mean(dep_yData),np.min(xData), np.max(dep_yData)]) # Initial deposition parameter guess
            sco_par, sco_intCurve, sco_covar, sco_params_interp =  interpolate(func_exp2, xData, sco_yData, ic=sco_ic, bounds=(-np.inf, np.inf))
        elif volume_func_mode == 3:
            #TODO check initial guess for this function
            sco_ic=np.array([np.mean(dep_yData),np.min(xData), np.max(dep_yData)]) # Initial deposition parameter guess
            sco_par, sco_intCurve, sco_covar, sco_params_interp =  interpolate(func_exp3, xData, sco_yData, ic=sco_ic, bounds=(-np.inf, np.inf))
            
        if i == 0:
            sco_params = sco_params_interp
        else:
            params = sco_params_interp
            sco_params  = np.row_stack((sco_params,params))
        
        
        # Morphological active width interpolation:
        morphWact_yData = int_d["int_morphWact_data_" + run][:,i]
        if morphWact_func_mode == 1:
            morphWact_ic=np.array([np.mean(dep_yData),np.min(xData)]) # Initial deposition parameter guess
            morphWact_par, morphWact_intCurve, morphWact_covar, morphWact_params_interp =  interpolate(func_exp, xData, morphWact_yData, ic=morphWact_ic, bounds=(-np.inf, np.inf))
        elif morphWact_func_mode ==2:
            morphWact_ic=np.array([np.mean(dep_yData),np.min(xData), np.max(dep_yData)]) # Initial deposition parameter guess
            morphWact_par, morphWact_intCurve, morphWact_covar, morphWact_params_interp =  interpolate(func_exp2, xData, morphWact_yData, ic=morphWact_ic, bounds=(-np.inf, np.inf))
        elif morphWact_func_mode == 3:
            #TODO check initial guess for this function
            morphWact_ic=np.array([np.mean(dep_yData),np.min(xData), np.max(dep_yData)]) # Initial deposition parameter guess
            morphWact_par, morphWact_intCurve, morphWact_covar, morphWact_params_interp =  interpolate(func_exp3, xData, morphWact_yData, ic=morphWact_ic, bounds=(-np.inf, np.inf))
                
        if i == 0:
            morphWact_params = morphWact_params_interp
        else:
            params = morphWact_params_interp
            morphWact_params = np.row_stack((morphWact_params,params))
            
        
        # Active thickness interpolation:
        act_thickness_yData = int_d["int_act_thickness_data_" + run][:,i]
        if act_thickness_func_mode == 1:
            act_thickness_ic=np.array([np.mean(act_thickness_yData),np.min(xData)]) # Initial deposition parameter guess
            act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
        elif act_thickness_func_mode ==2:
            act_thickness_ic=np.array([np.mean(act_thickness_yData),np.min(xData), np.max(act_thickness_yData)]) # Initial deposition parameter guess
            act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp2, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
        elif act_thickness_func_mode == 3:
            #TODO check initial guess for this function
            act_thickness_ic=np.array([np.mean(act_thickness_yData),np.min(xData), np.max(act_thickness_yData)]) # Initial deposition parameter guess
            act_thickness_par, act_thickness_intCurve, act_thickness_covar, act_thickness_params_interp =  interpolate(func_exp3, xData, act_thickness_yData, ic=act_thickness_ic, bounds=(-np.inf, np.inf))
                
        if i == 0:
            act_thickness_params = act_thickness_params_interp
        else:
            params = act_thickness_params_interp
            act_thickness_params = np.row_stack((act_thickness_params,params))
            
            
            print()
            print('RUN: ', run)
            print('Series: ', i)
            print('Scour interp A: ', sco_par[0], 'SD', sco_covar[0,0])
            print('Scour interp B: ', sco_par[1], 'SD', sco_covar[1,1])
            print()
        
        if plot_mode == 1:
            fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(xData, dep_yData, 'o', c='blue')
            axs.plot(xData, dep_intCurve, c='green')
            axs.set_title('Deposition series # '+str(i+1)+'- '+run)
            axs.set_xlabel('Time [min]')
            axs.set_ylabel('Volume [mm³]')
            plt.savefig(os.path.join(plot_dir, run + 'series_' + str(i+1) +'_dep_interp.png'), dpi=200)
            plt.show()
            
            fig2, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(xData, sco_yData, 'o', c='red')
            axs.plot(xData, sco_intCurve, c='green')
            axs.set_title('Scour series # '+str(i+1)+'- '+run)
            axs.set_xlabel('Time [min]')
            axs.set_ylabel('Volume [mm³]')
            plt.savefig(os.path.join(plot_dir, run + 'series_' + str(i+1) +'_sco_interp.png'), dpi=200)
            plt.show()
            
            fig3, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(xData, morphWact_yData, 'o', c='brown')
            axs.plot(xData, morphWact_intCurve, c='green')
            axs.set_title('Morphological active width series # '+str(i+1)+'- '+run)
            axs.set_xlabel('Time [min]')
            axs.set_ylabel('morphWact [mm³]')
            plt.savefig(os.path.join(plot_dir, run + 'series_' + str(i+1) +'_morphWact_interp.png'), dpi=200)
            plt.show()
            
            fig4, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True)
            axs.plot(xData, act_thickness_yData, 'o', c='purple')
            axs.plot(xData, act_thickness_intCurve, c='green')
            axs.set_title('Active thickness # '+str(i+1)+'- '+run)
            axs.set_xlabel('Time [min]')
            axs.set_ylabel('Active thickness [mm]')
            plt.savefig(os.path.join(plot_dir, run + 'series_' + str(i+1) +'_act_thickness_interp.png'), dpi=200)
            plt.show()
            
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
    
    # Print txt reports:
    header = 'Serie1, Serie2, Serie3, Serie4' # Report header
    np.savetxt(os.path.join(int_report_dir, run + '_dep_int_param.txt'), dep_params, delimiter=',', header = header)
    np.savetxt(os.path.join(int_report_dir, run + '_sco_int_param.txt'), dep_params, delimiter=',', header = header)
    np.savetxt(os.path.join(int_report_dir, run + '_morphWact_int_param.txt'), morphWact_params, delimiter=',', header = header)
    np.savetxt(os.path.join(int_report_dir, run + '_act_thickness_int_param.txt'), act_thickness_params, delimiter=',', header = header)  

end = time.time()
print()
print('Execution time: ', (end-start), 's')