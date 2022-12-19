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
    par, covar = opt.curve_fit(func, xData, yData, p0=ic, maxfev=80000, bounds=bounds)
    pars=[]
    if len(par) == 1:
        intCurve = func(xData, par[0])
    elif len(par) == 2:
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
def func_exp(x,B):
    # func_mode = 1
    y = 1*(1-np.exp(-x/B))
    return y

def func_exp2(x,A,B,C):
    # func_mode = 2
    y = C + A*(1-np.exp(-x/B))
    return y

# morphW interpolation function:
def func_exp3(x,A,B):
    # func_mode = 3
    y = A + (1-A)*(1-np.exp(-x/B))
    return y

def func_exp4(x,A,B):
    # func_mode = 4
    y = 2 + A*(1-np.exp(-x/B))
    return y

def func_exp5(x,A,B):
    # func_mode = 1
    y = A*(1-np.exp(-x/B))
    return y

###############################################################################
# SCRIPT PARAMETERS
###############################################################################
start = time.time()


font = {'family': 'serif',
        'color':  'black',
        'weight': 'regular',
        'size': 12
        }

###############################################################################
# SETUP FOLDERS
###############################################################################

w_dir = os.getcwd() # set working directory
data_folder = os.path.join(w_dir, 'interpolation_script', 'input_data') # Set path were source data are stored
plot_dir = os.path.join(w_dir, 'interpolation_script', 'output_data', 'interpolation_plot')
int_report_dir = os.path.join(w_dir, 'interpolation_script', 'output_data', 'int_report')
if not(os.path.exists(plot_dir)):
    os.mkdir(plot_dir)
if not(os.path.exists(int_report_dir)):
    os.mkdir(int_report_dir)


# EXTRACT ALL RUNS
# RUNS = ['q07_1', 'q10_2', 'q15_2', 'q20_2'] # Initialize RUS array
RUNS = ['q07_1'] # Initialize RUS array


###############################################################################
# Loop over all runs
###############################################################################
d={}
d_int={}

for run in RUNS:
    print('*************')
    print(run)
    print('*************')
    # Extract the number of survey files in the survey folder
    # Needed to know the dimension of the matrix
    
    ###########################################################################
    # LOAD PARAMETERS MATRIX
    # discharge [l/s],repetition,run time [min],Texner discretization, Channel width [m], slome [m/m]
    parameters = np.loadtxt(os.path.join(w_dir, 'parameters.txt'),
                            delimiter=',',
                            skiprows=1)
    # Extract run parameter
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,1]==float(run[-1:])),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]

    dt = run_param[0,2] # dt between runs in minutes (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    W = run_param[0,4] # Flume width [m]
    
    ###########################################################################
    # Create the d dictionary entry loading report files from input_data dir
    d["MAW_envelope_{0}".format(run)] = np.loadtxt(os.path.join(data_folder, 'envelope_area_data.txt'), delimiter = ',', skiprows=2)
    

    ###########################################################################
    # CREATE DICTIONARY ENTRIES WITH ZEROS
    # d_int["int_sum_vol_data_{0}".format(run)] = np.zeros((N, N))

    

    # Build up x_data array
    
    xData = np.arange(1, d["MAW_envelope_{0}".format(run)].shape[0]+1, 1)*dt # Time in Txnr
    
    yData_area_envelope = d["MAW_envelope_{0}".format(run)][:,1]
    
    
    ###########################################################################
    #   INTERPOLATION
    
    # Interpolation data initial guess
    # interpolation_ic = np.array([np.min(yData_area_envelope)/2, np.mean(yData_area_envelope)])
    interpolation_ic = np.array([np.min(xData)])
    
    yData_mean_par, yData_mean_intCurve, yData_mean_covar, yData_mean_params_interp =  interpolate(func_exp, xData, yData_area_envelope, ic=interpolation_ic, bounds=(-np.inf, np.inf))
    # yData_mean_par, yData_mean_intCurve, yData_mean_covar, yData_mean_params_interp =  interpolate(func_exp3, xData, yData_area_envelope, bounds=(-np.inf, np.inf))
    
    
    
    # PLOT THE DATA AND THE INTERPOLATION CURVE FOR THE MEAN DATASET
        
    # Deposition volume plot - MEAN
    fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
    axs.plot(xData, yData_area_envelope, 'o', c='blue')
    axs.plot(xData, yData_mean_intCurve, c='green')
    axs.plot(xData, 1/yData_mean_params_interp[0]*np.exp(-xData/yData_mean_params_interp[0]))
    axs.set_title('MAW envelope interpolation - ' +run)
    axs.set_xlabel('Time [min]')
    axs.set_ylabel('MAW/W [-]')
    axs.set_ylim(bottom=0)
    # axs.set_xlim(bottom=0)
    # plt.text(np.max(xData)*0.7, np.min(yData_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(yData_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(yData_mean_params_interp[0], decimals=1)), fontdict=font)
    plt.text(np.max(xData)*0.7, np.min(yData_area_envelope), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(yData_mean_params_interp[0], decimals=1)) + 'min \n' , fontdict=font)
    plt.savefig(os.path.join(plot_dir, run  + 'MAA_envelope_interp.pdf'), dpi=200)
    plt.savefig(os.path.join(plot_dir, run  + 'MAA_envelpe_interp.png'), dpi=300)
    plt.show()
    
    yData_derivate = 1/yData_mean_params_interp[0]*np.exp(-xData/yData_mean_params_interp[0])
    
    # Deposition volume plot - MEAN
    fig2, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(6,4))
    # axs.plot(xData, yData_area_envelope, 'o', c='blue')
    # axs.plot(xData, yData_mean_intCurve, c='green')
    axs.plot(xData, yData_derivate, c='red') # Plot the derivate of the exp function
    axs.set_title('MAW envelope interpolation - ' +run)
    axs.set_xlabel('Time [min]')
    axs.set_ylabel('MAW/W [-]')
    axs.set_ylim(bottom=0)
    # axs.set_xlim(bottom=0)
    # plt.text(np.max(xData)*0.7, np.min(yData_mean_intCurve), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(yData_mean_params_interp[2], decimals=1)) + 'min \n' + 'A = ' + str(np.round(yData_mean_params_interp[0], decimals=1)), fontdict=font)
    plt.text(np.min(xData)*0.7, np.min(yData_derivate), 'Trun=' + str(dt) + 'min \n' + r'$\tau$=' + str(np.round(yData_mean_params_interp[0], decimals=1)) + 'min \n' , fontdict=font)
    plt.savefig(os.path.join(plot_dir, run  + 'MAA_envelope_interp.pdf'), dpi=200)
    plt.savefig(os.path.join(plot_dir, run  + 'MAA_envelpe_interp.png'), dpi=300)
    plt.show()


    # Transpose arrays to obtain structure as below:
    #        Serie1    Serie2    Serie3    Serie4
    #   A    
    # SD(A)
    #   B
    # SD(B)
    #   C
    # SD(C)
    
    # dep_params = np.transpose(dep_params)
    # sco_params = np.transpose(sco_params)
    # morphWact_params = np.transpose(morphWact_params)
    # act_thickness_params = np.transpose(act_thickness_params)
    # act_area_params = np.transpose(act_area_params)
    
    # # Print txt reports:
    # header = 'Serie1, Serie2, Serie3, Serie4' # Report header
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_dep_int_param.txt'), dep_params, delimiter=',', header = header)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_sco_int_param.txt'), sco_params, delimiter=',', header = header)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(morphWact_func_mode) + '_morphWact_int_param.txt'), morphWact_params, delimiter=',', header = header)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_thickness_func_mode) + '_act_thickness_int_param.txt'), act_thickness_params, delimiter=',', header = header)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_area_func_mode) + '_act_area_int_param.txt'), act_area_params, delimiter=',', header = header)
    
    # # Print txt reports for interpolation mean:
    # header_mean = 'Interpolation parameters of the mean value of each run'
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_dep_int_param_mean.txt'), dep_mean_params_interp, delimiter=',', header = header_mean)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(volume_func_mode) + '_sco_int_param_mean.txt'), sco_mean_params_interp, delimiter=',', header = header_mean)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(morphWact_func_mode) + '_morphWact_int_param_mean.txt'), morphWact_mean_params_interp, delimiter=',', header = header_mean)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_thickness_func_mode) + '_act_thickness_int_param_mean.txt'), act_thickness_mean_params_interp, delimiter=',', header = header_mean)
    # np.savetxt(os.path.join(int_report_dir, run + '_func_mode' + str(act_area_func_mode) + '_act_area_int_param_mean.txt'), act_area_mean_params_interp, delimiter=',', header = header_mean)
    
    
end = time.time()
print()
print('Execution time: ', (end-start), 's')