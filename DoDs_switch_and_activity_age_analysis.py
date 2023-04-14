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

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time


# SINGLE RUN NAME
# runs = ['q07_1']
runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']

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
    def trim_consecutive_equal(arr):
        # Make sure the array is a numpy array
        arr = np.array(arr)
        
        # Create an array that indicates whether each value is equal to the next
        equal_to_next = np.hstack((arr[:-1] == arr[1:], False))
    
        # Use this to mask out the consecutive equal values
        trimmed_arr = arr[~equal_to_next]
        
        return trimmed_arr
    
    
    d0_slice = np.array(stack[:, :, :, delta])
    d0_slice = np.where(d0_slice>0, 1, d0_slice)
    d0_slice = np.where(d0_slice<0, -1, d0_slice)
    
    mask=d0_slice[0,:,:]
    mask=np.where(np.logical_not(np.isnan(mask)), 1, np.nan)
    
    switch_matrix = np.zeros((dim_y, dim_x))
    
    for i in range(0,dim_y):
        for j in range(0,dim_x):
            
            d0 = d0_slice[:,i,j]
            
            # Test array
            # d0 = np.array([0,np.nan,0,1,0,1,0,-1,1,-1,0,0,-1,0])
            
            d0 = d0[d0!=0] # Trim zero values
            
            d0 = d0[np.logical_not(np.isnan(d0))]
            
            if len(d0) == 0:
                switch = 0
            else:
                d0 = trim_consecutive_equal(d0)
                switch = int(len(d0)-1)
            
            switch_matrix[i,j] = switch
            

    # PLOT
    # Plot the results as a heatmap
    plt.imshow(switch_matrix, cmap='Reds', aspect=0.1)
    plt.title('number of switch - ' + run)
    plt.colorbar(orientation='horizontal')
    plt.savefig(os.path.join(report_dir, run,run +'_switch_spatial_distribution.pdf'), dpi=300)
    plt.show()
    
    switch_matrix = switch_matrix*mask
    
    switch_matrix = switch_matrix[np.logical_not(np.isnan(switch_matrix))]
    
    hist_switch, bins_switch = np.histogram(switch_matrix.flatten(), bins=range(int(np.min(switch_matrix.flatten())), int(np.max(switch_matrix.flatten())+2)), density=True)

    # plot the histogram
    plt.bar(bins_switch[:-1], hist_switch, align='center', alpha=0.5, label='Activity')
    plt.xticks(bins_switch[:-1])
    plt.legend()
    plt.ylim(0, 0.8)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Number of switch - ' + run)
    plt.savefig(os.path.join(report_dir,run +'_number_of_switch.pdf'), dpi=300)
    plt.show()
    