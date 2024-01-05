#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:02:28 2023

@author: erri

This script analyze the DoD at a give delta timespan and compute:
    1. The number of switch that occours during the run and provieds a map
    of the spatially distributed number of switch
    2. Compute the frequency distribution of the number of switch

NB: This script do not includes zero in the switch calculation:
    so [1,0,1] is not considered as a switch between 1 and 0.
    

"""

# IMPORT PACKAGES
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from period_function_v3 import *


# VERY FIRST SETUP
start = time.time() # Set initial time

plot_mode=[
    'periods_dist',
    'switch_number_dist'
    ]

# SINGLE RUN NAME
# runs = ['q07_1']
# runs = ['q10_2']
# runs = ['q10_3']
# runs = ['q10_4']
# runs = ['q15_2']
# runs = ['q15_3']
# runs = ['q20_2']
runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q10_4', 'q15_2', 'q15_3', 'q20_2']

# DoD timespan
t_span = 3

'''
INPUT:
    the script takes in input a stack where all the DoDs are stored in a structure as shown below:
        
    DoD input stack structure:
        
        DoD_stack[time,y,x,delta]
        DoD_stack_bool[time,y,x,delta]
        
         - - - 0 - - - - 1 - - - - 2 - - - - 3 - - - - 4 - - - - 5 - - - - 6 - - - - 7 - - - - 8 - -  >    delta
      0  |  DoD 1-0   DoD 2-0   DoD 3-0   DoD 4-0   DoD 5-0   DoD 6-0   DoD 7-0   DoD 8-0   DoD 9-0
      1  |  DoD 2-1   DoD 3-1   DoD 4-1   DoD 5-1   DoD 6-1   DoD 7-1   DoD 8-1   DoD 9-1
      2  |  DoD 3-2   DoD 4-2   DoD 5-2   DoD 6-2   DoD 7-2   DoD 8-2   DoD 9-2
      3  |  DoD 4-3   DoD 5-3   DoD 6-3   DoD 7-3   DoD 8-3   DoD 9-3
      4  |  DoD 5-4   DoD 6-4   DoD 7-4   DoD 8-4   DoD 9-4
      5  |  DoD 6-5   DoD 7-5   DoD 8-5   DoD 9-5
      6  |  DoD 7-6   DoD 8-6   DoD 9-6
      7  |  DoD 8-7   DoD 9-7
      8  |  DoD 9-8
         |
         v
        
         time
        
'''
# INITIALIZE ARRAY

      
for run in runs:
    print(run, ' is running...')
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
    output_dir = os.path.join(report_dir, 'report_'+run, 'switch_analysis')
    plot_dir = os.path.join(output_dir, 'plot')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)
        
    # INITIALIZE ARRAYS

    
    ###########################################################################

    # IMPORT DoD STACK AND DoD BOOL STACK
    DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    # Load diff stack for new activated or deactivated area
    # 1=activated, -1=deactivated, 0=no_changes
    diff_stack = np.load(os.path.join(home_dir, 'output','report_'+run, run + '_diff_stack.npy'))
    
    if t_span==0:
        diff_stack = diff_stack[:,:,:,t_span]
        stack_bool = stack_bool[:,:,:,t_span]
    else:
        diff_stack = diff_stack[:-t_span,:,:,t_span]
        stack_bool = stack_bool[:-t_span,:,:,t_span]

    dim_t, dim_y, dim_x, dim_d = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    # COMPUTE THE ACTIVE AREA
    act_stack = np.abs(stack_bool[:,:,:])
    activity_arr   = np.nansum(act_stack, axis=2)
    activity_arr   = np.nansum(activity_arr, axis=1)
    mean_activity_area = np.nanmean(activity_arr)
    
    # COMPUTE AND STORE THE ACTIVATED AND DEACTIVATED NUMBER OF PIXEL
    diff_stack_act = diff_stack*(diff_stack>0) # Extract activated pixel only
    diff_stack_deact = diff_stack*(diff_stack<0) # Extract deactivated pixel only
    
    act_arr           = np.nansum(diff_stack_act, axis=2)
    act_arr           = np.nansum(act_arr, axis=1)
    act_arr_rel       = act_arr/activity_arr[:-1]
    act_arr_abs       = act_arr/dim_x/120
    act_arr_rel_mean  = np.nanmean(act_arr_rel)
    act_arr_abs_stdev = np.nanstd(act_arr_rel)
    
    deact_arr           = np.nansum(diff_stack_deact, axis=2)
    deact_arr           = np.nansum(deact_arr, axis=1)
    deact_arr_rel       = deact_arr/activity_arr[:-1]
    deact_arr_abs       = deact_arr/dim_x/120
    deact_arr_rel_mean  = np.nanmean(deact_arr_rel)
    deact_arr_abs_stdev = np.nanstd(deact_arr_rel)
    
    print(act_arr_rel)
    print(act_arr_abs)
    
    # Save report as txt file
    np.savetxt(os.path.join(output_dir, run+'_relative_activated_pixels.txt'), act_arr_rel, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, run+'_relative_deactivated_pixel.txt'), deact_arr_rel, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, run+'_absolute_activated_pixels.txt'), act_arr_abs, fmt='%.4f')
    np.savetxt(os.path.join(output_dir, run+'_absolute_deactivated_pixel.txt'), deact_arr_abs, fmt='%.4f')
    
    

    
    ###########################################################################
    # PLOTS
    ###########################################################################
    
        