#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:01:06 2023

@author: erri

Taking advantages of the period_length punction the script compute the length of
a consecutive series of scour (-1), fill (+1) and no changes (0) along the time
axis of aa DoDs stack.
The script provides:
    1. Histograms for each runds of the period length


"""

# IMPORT PACKAGES
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

delta = 0

# SINGLE RUN NAME
runs = ['q07_1']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q15_2', 'q15_3', 'q20_2']

def period_length(arr):
    '''
    Given a numpy array the function calculate the number of consecutive 1, 0 and -1
    and returns the length of each periods in three different array.

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.

    Returns
    -------
    consec_1_arr : TYPE
        DESCRIPTION.
    consec_0_arr : TYPE
        DESCRIPTION.
    consec_neg1_arr : TYPE
        DESCRIPTION.

    '''
    # initialize variables
    consec_1 = 0
    consec_0 = 0
    consec_neg1 = 0
    
    # initialize lists to collect results
    consec_1_list = []
    consec_0_list = []
    consec_neg1_list = []
    
    # iterate over array elements
    for val in arr:
        if val == 1:
            consec_1 += 1
            if consec_0 > 0:
                consec_0_list.append(consec_0)
                consec_0 = 0
            if consec_neg1 > 0:
                consec_neg1_list.append(consec_neg1)
                consec_neg1 = 0
        elif val == 0:
            consec_0 += 1
            if consec_1 > 0:
                consec_1_list.append(consec_1)
                consec_1 = 0
            if consec_neg1 > 0:
                consec_neg1_list.append(consec_neg1)
                consec_neg1 = 0
        elif val == -1:
            consec_neg1 += 1
            if consec_1 > 0:
                consec_1_list.append(consec_1)
                consec_1 = 0
            if consec_0 > 0:
                consec_0_list.append(consec_0)
                consec_0 = 0
    
    # append any remaining values
    if consec_1 > 0:
        consec_1_list.append(consec_1)
    if consec_0 > 0:
        consec_0_list.append(consec_0)
    if consec_neg1 > 0:
        consec_neg1_list.append(consec_neg1)
    
    # convert lists to numpy arrays
    consec_1_arr = np.array(consec_1_list)
    consec_0_arr = np.array(consec_0_list)
    consec_neg1_arr = np.array(consec_neg1_list)
    return     consec_1_arr, consec_0_arr, consec_neg1_arr

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
    
    # stack_bool = np.where(np.isnan(stack_bool), 0, stack_bool)
    stack_bool=stack_bool[:,:,:,delta]
    
    dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    




    # # example numpy array to test teh function
    # arr = np.array([ 1,  1,  1, -1, -1, -1,  0,  0,  0,  0])
    # consec_1_arr, consec_0_arr, consec_neg1_arr = period_length(arr)
    # print("Consecutive 1s:", consec_1_arr)
    # print("Consecutive 0s:", consec_0_arr)
    # print("Consecutive -1s:", consec_neg1_arr)
    
    # stack = np.zeros((10, 5, 5), dtype=int)
    
    # # Fill one third of the stack with 1s
    # stack[:3, :, :] = 1
    
    # # Fill another third of the stack with -1s
    # stack[3:6, :, :] = -1
    
    consec_1_stack = np.zeros_like(stack_bool)
    consec_0_stack = np.zeros_like(stack_bool)
    consec_neg1_stack = np.zeros_like(stack_bool)
    
    for i in range(stack_bool.shape[1]):
        for j in range(stack_bool.shape[2]):
            
            sliced_arr = stack_bool[:, i, j]
            
            consec_1_arr, consec_0_arr, consec_neg1_arr = period_length(sliced_arr)
            
            consec_1_stack[0:len(consec_1_arr), i, j] = consec_1_arr
            consec_0_stack[0:len(consec_0_arr), i, j] = consec_0_arr
            consec_neg1_stack[0:len(consec_neg1_arr), i, j] = consec_neg1_arr
    
    
    
    
        
    flat_mat =consec_1_stack.flatten().astype(int)
    flat_mat = flat_mat[flat_mat!=0]

    # compute the histogram of values
    hist, bins = np.histogram(flat_mat, bins=range(np.min(flat_mat), np.max(flat_mat)+2), density=True)
    
    # plot the histogram
    plt.bar(bins[:-1], hist, align='center')
    plt.xticks(bins[:-1])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency distribution of +1 periods length - ' + run)
    plt.savefig(os.path.join(report_dir, run,run +'pos1_period_len_frequency_distribution.pdf'), dpi=200)
    plt.show()
    

    flat_mat =consec_0_stack.flatten().astype(int)
    flat_mat = flat_mat[flat_mat!=0]

    # compute the histogram of values
    hist, bins = np.histogram(flat_mat, bins=range(np.min(flat_mat), np.max(flat_mat)+2), density=True)
    
    # plot the histogram
    plt.bar(bins[:-1], hist, align='center')
    plt.xticks(bins[:-1])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency distribution of 0 periods length - ' + run)
    plt.savefig(os.path.join(report_dir, run,run +'zero_period_len_frequency_distribution.pdf'), dpi=200)
    plt.show()
    
    
    flat_mat =consec_neg1_stack.flatten().astype(int)
    flat_mat = flat_mat[flat_mat!=0]

    # compute the histogram of values
    hist, bins = np.histogram(flat_mat, bins=range(np.min(flat_mat), np.max(flat_mat)+2), density=True)
    
    # plot the histogram
    plt.bar(bins[:-1], hist, align='center')
    plt.xticks(bins[:-1])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency distribution of -1 periods length - ' + run)
    plt.savefig(os.path.join(report_dir, run,run +'neg1_period_len_frequency_distribution.pdf'), dpi=200)
    plt.show()
    
