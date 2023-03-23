#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:02:28 2023

@author: erri
"""

# IMPORT PACKAGES
import os
import cv2
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time


# SINGLE RUN NAME
runs = ['q07_1']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q15_2', 'q15_3', 'q20_2']

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
    d0_slice = stack[:, :, :, 0]
    
    # Compute the number of sign switches along the t-axis
    sign_changes = np.diff(np.sign(d0_slice), axis=0)
    num_changes = np.sum(sign_changes != 0, axis=0)
    
    # Plot the results as a heatmap
    plt.imshow(num_changes, cmap='hot')
    plt.colorbar()
    plt.show()
    
    #%%
    # Compute the sign of each element along the t-axis
    signs = np.sign(d0_slice)
    
    # Compute the differences between adjacent signs along the t-axis
    diffs = np.diff(signs, axis=0)
    
    # Find the indices where the sign changes occur
    change_indices = np.where(diffs != 0)
    
    # Compute the lengths of the positive and negative periods
    period_lengths = np.diff(change_indices, axis=1)
    
    # Extract the lengths of the positive and negative periods
    pos_lengths = period_lengths[0, np.where(diffs[change_indices] == 1)]
    neg_lengths = period_lengths[0, np.where(diffs[change_indices] == -1)]
    
    # Remove zero-length periods
    pos_lengths = pos_lengths[pos_lengths > 0]
    neg_lengths = neg_lengths[neg_lengths > 0]
    
    # Plot histograms of the length of positive and negative periods
    plt.hist(pos_lengths, bins=20, alpha=0.5, label='Positive')
    plt.hist(neg_lengths, bins=20, alpha=0.5, label='Negative')
    plt.legend()
    plt.xlabel('Length of period')
    plt.ylabel('Frequency')
    plt.show()
            
    
