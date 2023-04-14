#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:02:28 2023

@author: erri

This script analyze the DoD at a give delta timespan and compute:
    1. The number of switch that occours during the run and provieds a map
    of the spatially distributed number of switch
    2. Compute the frequency distribution of the number of switch

NB: This script do includes zero in the switch calculation:
    so [1,0,1] is  considered as two switch between 1 and 0 and then between 0 and 1.

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
    
    # 1. number of switch
    
    # Slice the stack to consider only d=0
    d0_slice = np.array(stack[:, :, :, delta])
    d0_slice = np.where(d0_slice>0, 1, d0_slice)
    d0_slice = np.where(d0_slice<0, -1, d0_slice)
    
    # Test array
    # d0_slice = np.array([0,0,1,1,1,1,0,-1,1,-1,0,0,-1,0])
    # Compute the number of sign switches along the t-axis
    
    # d0_slice = d0_slice[:,13,5]
    sign_changes = np.diff(np.sign(d0_slice), axis=0)
    num_changes = np.sum(sign_changes != 0, axis=0)
    
    
    
    
    print(num_changes)
    # Define a mask for out of domain values
    mask=d0_slice[0,:,:]
    mask=np.where(np.logical_not(np.isnan(mask)), 1, np.nan)
    
    
    # Plot the results as a heatmap
    plt.imshow(num_changes*mask, cmap='Reds', aspect=0.1)
    plt.title('Number of switch - ' + run)
    plt.colorbar(orientation='horizontal')
    plt.savefig(os.path.join(report_dir, run,run +'_switch_zero_included_spatial_distribution.pdf'), dpi=300)
    plt.show()
    
    
    
    
    # COMPUTE THE SWITCH FREQUENCY DISTRIBUTION OF EACH RUNS
    
    matrix=num_changes*mask
    matrix = matrix[np.logical_not(np.isnan(matrix))]
    
    hist_ones, bins_ones = np.histogram(matrix.flatten(), bins=range(int(np.min(matrix.flatten())), int(np.max(matrix.flatten())+2)), density=True)

    # plot the histogram
    plt.bar(bins_ones[:-1], hist_ones, align='center', alpha=0.5, label='Activity')

    plt.xticks(bins_ones[:-1])
    plt.legend()
    plt.ylim(0, 0.40)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Number of switch - ' + run)
    plt.savefig(os.path.join(report_dir,run +'_number_of_switch_zero_included.pdf'), dpi=300)
    plt.show()
