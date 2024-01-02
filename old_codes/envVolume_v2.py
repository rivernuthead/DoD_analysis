#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:38:09 2023

@author: erri

This script compute the volume of the envelop of DoDs at different timespan.

INPUT:
    the script takes in input a stack where all the DoDs are stored in a structure as shown below:
        DoD1-0  DoD2-0  DoD3-0  DoD4-0  DoD5-0  DoD6-0  DoD7-0  DoD8-0  DoD9-0
        DoD2-1  DoD3-1  DoD4-1  DoD5-1  DoD6-1  DoD7-1  DoD8-1  DoD9-1
        DoD3-2  DoD4-2  DoD5-2  DoD6-2  DoD7-2  DoD8-2  DoD9-2
        DoD4-3  DoD5-3  DoD6-3  DoD7-3  DoD8-3  DoD9-3
        DoD5-4  DoD6-4  DoD7-4  DoD8-4  DoD9-4
        DoD6-5  DoD7-5  DoD8-5  DoD9-5
        DoD7-6  DoD8-6  DoD9-6
        DoD8-7  DoD9-7
        DoD9-8
        
        stack = [h,:,:, delta]


OUTPUT:
    envVol_sum_report[i,j]
    envVol_sco_report[i,j]
    envVol_fill_report[i,j]
    vol_data_structure[i,j]
"""

import os
import numpy as np

set_name = 'q10_2'
# IMPORT DoD STACK AND DoD BOOL STACK
home_dir = os.getcwd() # Home directory
DoDs_folder = os.path.join(home_dir, 'output', 'DoDs', 'DoDs_stack') # Input folder
stack_name = 'DoD_stack' + '_' + set_name + '.npy' # Define stack name
stack_bool_name = 'DoD_stack' + '_bool_' + set_name + '.npy' # Define stack bool name
stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path

stack = np.load(stack_path) # Load DoDs stack
stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack

dim_t, dim_y, dim_x, dim_delta = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension


# COMPUTE THE ENVELLOPE CONSIDERING THE VOLUME
envVol_sum_report  = np.zeros((dim_t,dim_t))
envVol_sco_report  = np.zeros((dim_t,dim_t))
envVol_fill_report = np.zeros((dim_t,dim_t))
vol_data_structure = np.empty((dim_t, dim_t), dtype='U100')
DoD1_list = []
DoD1_list_abs = []
# Loop over each 1-timespan DoD 
for k in range(0, dim_t):
    # Append the matrix to the list
    DoD1_list.append(stack[k,:,:,0]) # append in the list consecutive 1-timespan DoD
    DoD1_list_abs.append(abs(stack[k,:,:,0]))
    
    # Fill the report matrices with data from the volume of the single run
    envVol_sum_report[0,k] = np.nansum(abs(stack[k,:,:,0]))
    envVol_sco_report[0,k] = np.nansum(np.where(stack[k,:,:,0]<0, stack[k,:,:,0], 0))
    envVol_fill_report[0,k] = np.nansum(np.where(stack[k,:,:,0]>0, stack[k,:,:,0], 0))
    

# Fill data structure array with the text for the first row   
# Calculate the sums of adjacent matrices, three adjacent matrices, and so on
for i in range(1, len(DoD1_list)):
    for j in range(len(DoD1_list) - i+1):
        text = f"Volume computed over {j} and {j+1}:"
        vol_data_structure[0,j] = text

        
for i in range(1, len(DoD1_list)):
    for j in range(len(DoD1_list) - i):
        if i == 1:
            # Sum adjacent DoDs
            sum_mat = DoD1_list_abs[j] + DoD1_list_abs[j+1]
            # print(f"Volume computed over DoD {j} and {j+1}:")
            text = f"Volume computed over {j} and {j+2}:"
            vol_data_structure[i,j] = text
            
            # Sum adjacent scour DoDs
            sco_mat = np.where(DoD1_list[j]<0, DoD1_list[j], 0) + np.where(DoD1_list[j+1]<0,  DoD1_list[j+1], 0)
            
            # Sum adjacent fill DoDs
            fill_mat = np.where(DoD1_list[j]>0, DoD1_list[j], 0) + np.where(DoD1_list[j+1]>0,  DoD1_list[j+1], 0)
            
        else:
            # Sum i adjacent DoDs
            sum_mat = sum(DoD1_list_abs[j:j+i+1])
            # print(f"Volume computed over envelope of DoD {j} to {j+i}:")
            text = f"Volume computed over {j} to {j+2}:"
            vol_data_structure[i,j] = text
            
            mats = DoD1_list[j:j+i+1]
            
        
            # Sum adjacent fill DoDs
            fill_result = []
            for matrix in mats:
                mask = matrix > 0  # create a boolean mask of positive values
                positive_values = matrix[mask]  # apply the mask to get only positive values
                sum_positive = np.sum(positive_values)  # sum the positive values
                fill_result.append(sum_positive)
            fill_mat = fill_result
            
            # Sum adjacent fill DoDs
            sco_result = []
            for matrix in mats:
                mask = matrix < 0  # create a boolean mask of positive values
                negative_values = matrix[mask]  # apply the mask to get only positive values
                sum_negative = np.sum(negative_values)  # sum the positive values
                sco_result.append(sum_negative)
            sco_mat = sco_result
        
        envVol_sum_report[i,j] = np.nansum(sum_mat)
        envVol_sco_report[i,j] = np.nansum(sco_mat)
        envVol_fill_report[i,j] = np.nansum(fill_mat)
        

    '''
    Data Structure
    
    MAA envelope of DoD 0 and 1:	MAA envelope of DoD 1 and 2:	MAA envelope of DoD 2 and 3:	MAA envelope of DoD 3 and 4:	MAA envelope of DoD 4 and 5:	MAA envelope of DoD 5 and 6:	MAA envelope of DoD 6 and 7:	MAA envelope of DoD 7 and 8:    MAA envelope of DoD 8 and 9:
    MAA envelope of DoD 0 to 2:	    MAA envelope of DoD 1 to 3:	    MAA envelope of DoD 2 to 4: 	MAA envelope of DoD 3 to 5:	    MAA envelope of DoD 4 to 6:	    MAA envelope of DoD 5 to 7:	    MAA envelope of DoD 6 to 8:	
    MAA envelope of DoD 0 to 3:	    MAA envelope of DoD 1 to 4:    	MAA envelope of DoD 2 to 5:	    MAA envelope of DoD 3 to 6:	    MAA envelope of DoD 4 to 7:	    MAA envelope of DoD 5 to 8:		
    MAA envelope of DoD 0 to 4:	    MAA envelope of DoD 1 to 5:	    MAA envelope of DoD 2 to 6:	    MAA envelope of DoD 3 to 7:	    MAA envelope of DoD 4 to 8:			
    MAA envelope of DoD 0 to 5:	    MAA envelope of DoD 1 to 6:   	MAA envelope of DoD 2 to 7:	    MAA envelope of DoD 3 to 8:				
    MAA envelope of DoD 0 to 6:	    MAA envelope of DoD 1 to 7:	    MAA envelope of DoD 2 to 8:					
    MAA envelope of DoD 0 to 7:  	MAA envelope of DoD 1 to 8:						
    MAA envelope of DoD 0 to 8:		MAA envelope of DoD 1 to 9:
    MAA envelope of DoD 0 to 9:
        
    stack = [h,:,:, delta]
    
    '''
    
