#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:24:32 2023

@author: erri
"""

# COMPUTE THE MAA ENVELOPE
import os
import numpy as np

set_name = 'q20_2'
###############################################################################
# IMPORT DoD STACK AND DoD BOOL STACK
home_dir = os.getcwd() # Home directory
DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path

stack = np.load(stack_path) # Load DoDs stack
stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack

dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension

# CREATE THE LIST OF DoD

envMAW_report = np.zeros((dim_t-1,dim_y,dim_x,dim_t-1))
envMAW_sum_report = np.zeros((dim_t-1,dim_t-1))
data_structure = np.empty((dim_t-1, dim_t-1), dtype='U100')

DoD1_list = []

    
# Loop over each 1-timespan DoD 
for k in range(0, dim_t):
    # Append the matrix to the list
    DoD1_list.append(abs(stack[k,:,:,0])) # append in the list consecutive 1-timespan DoD


# Calculate the sums of adjacent matrices, three adjacent matrices, and so on
for i in range(1, len(DoD1_list)):
    for j in range(len(DoD1_list) - i):
        if i == 1:
            # Sum adjacent matrices
            sum_mat = DoD1_list[j] + DoD1_list[j+1]
            print(f"MAA envelope of DoD {j} and {j+1}:")
            text = f"MAA envelope of DoD {j} and {j+1}:"
            data_structure[i-1,j] = text
            
        else:
            # Sum i adjacent matrices
            sum_mat = sum(DoD1_list[j:j+i+1])
            print(f"MAA envelope of DoD {j} to {j+i}:")
            text = f"MAA envelope of DoD {j} to {j+i}:"
            data_structure[i-1,j] = text
        
        envMAW_report[i-1,:,:,j] = sum_mat
        envMAW_sum_report[i-1,j] = np.nansum(sum_mat>0)/(dim_x*dim_y)
        print(np.nansum(sum_mat)/(700*6295))
        print()