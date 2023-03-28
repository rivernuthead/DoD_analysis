#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:02:28 2023

@author: erri

This script analyze the DoD at a give delta timespan and compute:
    1. The number of switch that occours during the run and provieds a map
    of the spatially distributed number of switch
    2. The script compute the length of the consecutive active and not active
    periods and provides an histogram of the frequency distribution of the
    periods length.

"""

# IMPORT PACKAGES
import os
import numpy as np
import time
import matplotlib.pyplot as plt

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time


# SINGLE RUN NAME
# runs = ['q07_1']
runs = ['q07_1', 'q10_2', 'q10_3', 'q15_2', 'q15_3', 'q20_2']

# DoD delta in timespan
delta = 0
for run in runs:
    
    # ALENGTH OF THE ANALYSIS WINDOW IN NUMBER OF CELL
    analysis_window = 123 # Number of columns compared to the PiQs photo length
    
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    run_dir = os.path.join(home_dir, 'surveys')
    DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
    
    # if not(os.path.exists(os.path.join(report_dir, run,  'stack_analysis'))):
    #     os.mkdir(os.path.join(report_dir, run,  'stack_analysis'))

    
    ###############################################################################
    # IMPORT DoD STACK AND DoD BOOL STACK
    DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    # 1. number of switch
    
    # Slice the stack to consider only d=0
    d0_slice = np.array(stack[:, :, :, delta])
    d0_slice = np.where(d0_slice>0, 1, d0_slice)
    
    # Compute the number of sign switches along the t-axis
    sign_changes = np.diff(np.sign(d0_slice), axis=0)
    num_changes = np.sum(sign_changes != 0, axis=0)
    
    # Defina a mask for out of domain values
    mask=d0_slice[0,:,:]
    mask=np.where(np.logical_not(np.isnan(mask)), 1, np.nan)
    
    
    # Plot the results as a heatmap
    plt.imshow(num_changes*mask, cmap='Reds', aspect=0.1)
    plt.title('number of active/inactive switch - ' + run)
    plt.colorbar(orientation='horizontal')
    plt.savefig(os.path.join(report_dir, run,run +'_switch_spatial_distribution.pdf'), dpi=300)
    plt.show()
    
    # # Compute the length of consecutive active or non-active periods
    # cons_ones = np.zeros_like(d0_slice)
    # cons_zeros = np.zeros_like(d0_slice)
    
    # for i in range(0,d0_slice.shape[1]):
    #     for j in range(0,d0_slice.shape[2]):
    #         # # Example input array
    #         # arr = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0 ,0])
    #         arr = d0_slice[:,i,j]
            
    #         # Compute the indices where the values in arr change
    #         idx = np.flatnonzero(np.diff(arr, prepend=-1, append=-1))
            
    #         # Compute the consecutive counts for each value in arr
    #         consec_counts = np.diff(idx)
            
    #         # Collect the results for each value separately
    #         ones_count = consec_counts[arr[idx[:-1]] == 1]
    #         zeros_count = consec_counts[arr[idx[:-1]] == 0]
            
    #         cons_ones[:len(ones_count),i,j]   = ones_count[:]
    #         cons_zeros[:len(zeros_count),i,j] = zeros_count[:]
            
    #         # # Output the results
    #         # print("Consecutive ones counts:", ones_count)
    #         # print("Consecutive zeros counts:", zeros_count)

    
    # # Trim zero values form matrix
    # cons_ones  = cons_ones[cons_ones!=0]
    # cons_zeros = cons_zeros[cons_zeros!=0]
    
    
    # hist_ones, bins_ones = np.histogram(cons_ones.flatten(), bins=range(int(np.min(cons_ones.flatten())), int(np.max(cons_ones.flatten())+2)), density=True)
    # hist_zeros, bins_zeros = np.histogram(cons_zeros.flatten(), bins=range(int(np.min(cons_zeros.flatten())), int(np.max(cons_zeros.flatten())+2)), density=True)
    
    # # plot the histogram
    # plt.bar(bins_ones[:-1], hist_ones, align='center', alpha=0.5, label='Activity')
    # plt.bar(bins_zeros[:-1], hist_zeros, align='center', alpha=0.5, label='Inactivity')
    # plt.xticks(bins_ones[:-1])
    # plt.xticks(bins_zeros[:-1])
    # plt.legend()
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Frequency distribution of activity and inactivity periods length - ' + run)
    # plt.savefig(os.path.join(report_dir, run,run +'zero_period_len_frequency_distribution.pdf'), dpi=300)
    # plt.show()


