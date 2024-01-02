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

#%%############################################################################
# VERY FIRST SETUP
start = time.time() # Set initial time

run_mode=[
    # 'plot',
    ]

# SINGLE RUN NAME
runs = ['q07_1']
# runs = ['q10_2']
# runs = ['q10_3']
# runs = ['q10_4']
# runs = ['q15_2']
# runs = ['q15_3']
# runs = ['q20_2']
# runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']
# runs = ['q07_1', 'q10_2', 'q10_3', 'q10_4', 'q15_2', 'q15_3', 'q20_2']

# DoD timespan
t_span = 0

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
        
for run in runs:
    
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
    output_dir = os.path.join(report_dir, 'report_'+run, 'switch_analysis')
    if not(os.path.exists(output_dir)):
        os.mkdir(output_dir)
        
    # INITIALIZE ARRAYS
    switch_AW_array = []

    
    ###############################################################################
    # IMPORT DoD STACK AND DoD BOOL STACK
    DoDs_folder = os.path.join(home_dir,'output', 'DoDs', 'DoDs_stack') # Input folder
    stack_name = 'DoD_stack' + '_' + run + '.npy' # Define stack name
    stack_bool_name = 'DoD_stack' + '_bool_' + run + '.npy' # Define stack bool name
    stack_path = os.path.join(DoDs_folder,stack_name) # Define stack path
    stack_bool_path = os.path.join(DoDs_folder,stack_bool_name) # Define stack bool path
    
    stack = np.load(stack_path) # Load DoDs stack
    stack_bool = np.load(stack_bool_path) # Load DoDs boolean stack
    
    dim_t, dim_y, dim_x, dim_d = stack.shape # Define time dimension, crosswise dimension and longitudinal dimension
    
    
    # LOOP FOR EACH TIMESPAN
    for t_span in range(stack.shape[3]):
        
        # Cut the stack given the timespan
        if t_span == 0:
            stack_bool_data = np.array(stack_bool[:,:,:,t_span])
        else:
            stack_bool_data = np.array(stack_bool[:-t_span, :, :, t_span])
            
        # DoD mask
        mask=stack_bool_data[0,:,:]
        mask=np.where(np.logical_not(np.isnan(mask)), 1, np.nan)
        
        # INITIALIZE THE ARRAY
        switch_matrix = np.zeros((dim_y, dim_x))
        
        for i in range(0,dim_y):
            for j in range(0,dim_x):
                
                stack_slice = stack_bool_data[:,i,j]
                
                
                d0 = d0[d0!=0] # Trim zero values
                
                d0 = d0[np.logical_not(np.isnan(d0))]
                
                if len(d0) == 0:
                    switch = 0
                else:
                    d0 = trim_consecutive_equal(d0)
                    switch = int(len(d0)-1)
                
                switch_matrix[i,j] = switch
                
        
        # SAVE SWITCH MATRIX AS TXT FILE
        np.savetxt(os.path.join(output_dir,run+'_delta'+str(delta)+'_switch_spatial_distribution.txt'), switch_matrix)
        
        #  COMPUTE THE EQUIVALENT SWITCH WIDTH (the portion of the total width that experienced switch)
        switch_matrix_bool = np.where(switch_matrix>0,1,0)
        switch_AW = np.nansum(switch_matrix_bool)/(switch_matrix.shape[1]*120)
        
        switch_AW_array = np.append(switch_AW_array, np.round(switch_AW, decimals=4)) # In this array are stored the switch_AW for incrising delta
        np.savetxt(os.path.join(output_dir,run+'_switch_AW_array.txt'), np.round(switch_AW_array, decimals=4))
        
        
        if 'plot' in run_mode:
            # PLOT
            # Plot the results as a heatmap
            plt.imshow(switch_matrix, cmap='Reds', aspect=0.1)
            plt.title('number of switch - ' + run)
            plt.colorbar(orientation='horizontal')
            plt.savefig(os.path.join(output_dir,run+'_delta'+str(delta) +'_switch_spatial_distribution.pdf'), dpi=300)
            plt.show()
            
            switch_data = switch_matrix*mask
            
            switch_data = switch_data[np.logical_not(np.isnan(switch_data))]
            
            hist_switch, bins_switch = np.histogram(switch_data.flatten(), bins=range(int(np.min(switch_data.flatten())), int(np.max(switch_data.flatten())+2)), density=True)
        
            # plot the histogram
            plt.bar(bins_switch[:-1], hist_switch, align='center', alpha=0.5, label='Activity')
            plt.xticks(bins_switch[:-1])
            plt.legend()
            plt.ylim(0, 0.8)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Number of switch - ' + run)
            plt.savefig(os.path.join(output_dir,run+'_delta'+str(delta) +'_number_of_switch.pdf'), dpi=300)
            plt.show()
    
    
    # #%%
    # '''
    # In this section...
    # '''
    
    # data = np.copy(stack_bool)
    
    # delta=0
    
    # data = data[:,:,:,delta]
    
    # switch_counter_report = np.zeros((dim_t,2))
    # switch_counter_matrix = np.zeros((dim_y,dim_x))
    
    # # Outer loop: Controls the size of adjacent elements
    # for size in range(2, data.shape[0] + 1):
    #     print(f"Size {size}:")
    
    #     # Inner loop: Iterates through the array to extract adjacent elements
    #     for i in range(data.shape[0] - size + 1):
    #         sub_data = data[i:i+size,:,:]
        
    #         switch_counter = []
    #         for i in range(0,dim_y):
    #             for j in range(0,dim_x):
    #                 d0 = sub_data[:,i,j]
    #                 d0 = d0[d0!=0] # Trim zero values
    #                 d0 = d0[np.logical_not(np.isnan(d0))]
    #                 if len(d0) == 0:
    #                     switch = 0
    #                 else:
    #                     d0 = trim_consecutive_equal(d0)
    #                     switch = int(len(d0)-1)
                        
    #                     switch_counter_matrix[i,j] = switch
                        
    #         switch_counter_matrix_bool = (switch_counter_matrix>0)*1
    #         switch_AW = np.nansum(switch_counter_matrix_bool)/(dim_x*120) 
            
    #         print(switch_AW)
            
    #         # switch_counter = switch_counter[switch_counter!=0]
            
    #         # switch_counter_report[size-2,0] = np.nanmean(switch_counter)
    #         # switch_counter_report[size-2,1] = np.nanstd(switch_counter)
    
    # # np.savetxt(os.path.join(output_dir, run+'_switch_counter_increasing_windows_dim.txt'), switch_counter_report, header='mean, stdev', delimiter=',')