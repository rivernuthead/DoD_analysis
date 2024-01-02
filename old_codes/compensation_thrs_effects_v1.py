#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:11:51 2022

@author: erri

Pixel age analysis over stack DoDs

INPUT (as .npy binary files):
    DoD_stack1 : 3D numpy array stack
        Stack on which DoDs are stored as they are, with np.nan
    DoD_stack1_bool : 3D numpy array stack
        Stack on which DoDs are stored as -1, 0, +1 data, also with np.nan
OUTPUTS:
    
    
"""
# Import
import os
import numpy as np
# import math
# import random
import time
import imageio
import matplotlib.pyplot as plt
from windows_stat_func import windows_stat
# from matplotlib import colors
# from matplotlib.ticker import PercentFormatter
# import seaborn as sns

#%%
start = time.time() # Set initial time

# SCRIPT MODE
'''
run mode:
    0 = runs in the runs list
    1 = one run at time
    2 = bath process 'all the runs in the folder'
plot_mode:
    ==0 plot mode OFF
    ==1 plot mode ON
'''
run_mode = 1
plot_mode = 1

delta = 1 # Delta time of the DoDs


# ARRAY OF RUNS
runs = ['q07_1', 'q10_2', 'q15_3', 'q20_2']

# runs = ['q07_1']
# runs = ['q10_2']
# runs = ['q15_3']
# runs = ['q20_2']


# Create the figure and axis
fig, ax = plt.subplots()

for run in runs:
    # FOLDER SETUP
    home_dir = os.getcwd() # Home directory
    report_dir = os.path.join(home_dir, 'output')
    run_dir = os.path.join(home_dir, 'surveys')
    DoDs_folder = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Input folder
    
    print('###############\n' + '#    ' + run + '    #' + '\n###############')
     
    #%%############################################################################
    '''
    This section calculate the envelop of an increasing number of 1-time-step DoD
    and extract the DoD in which the 1-time-step are contained:
    For example it calculate the envelope of DoD1-0, DoD2-1 and DoD3-2 and it also
    extract the DoD3-0.
    The envelop and the averall DoD maps were then translate as boolean activity map
    where 1 means activity and 0 means inactivity.
    Making the difference between this two maps (envelope - DoD), the diff map is a
    map of 1, -1 and 0.
    1 means that a pixel is active in the 1-time-step DoD evelope but is not detected
    as active in the overall DoD, so it is affcted by compensation processes.
    -1 means that a pixel is detected as active in the overall DoD but not in the
    1-time-step DoD so in the 1-time-step DoD it is always under the detection
    threshold (2mm for example).
    So this map provide us 2-dimensional information about pixel that experienced
    compensation or under thrashold processes.
    
    '''
    stack_dir = os.path.join(home_dir, 'DoDs', 'DoDs_stack') # Define the stack directory
    stack=np.load(os.path.join(stack_dir, 'DoD_stack_'+run+'.npy')) # Load the stack
    stack_bool=np.load(os.path.join(stack_dir, 'DoD_stack_bool_'+run+'.npy'))
    
        # Create output matrix as below:
        #            t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9
        # delta = 1  1-0   2-1   3-2   4-3   5-4   6-5   7-6   8-7   9-8  average STDEV
        # delta = 2  2-0   3-1   4-2   5-3   6-4   7-5   8-6   9-7        average STDEV
        # delta = 3  3-0   4-1   5-2   6-3   7-4   8-5   9-6              average STDEV
        # delta = 4  4-0   5-1   6-2   7-3   8-4   9-5                    average STDEV
        # delta = 5  5-0   6-1   7-2   8-3   9-4                          average STDEV
        # delta = 6  6-0   7-1   8-2   9-3                                average STDEV
        # delta = 7  7-0   8-1   9-2                                      average STDEV
        # delta = 8  8-0   9-1                                            average STDEV
        # delta = 9  9-0                                                  average STDEV
        
        # stack[time, x, y, delta]
    
    # # Initialize arrays:
    # comp_matrix = np.zeros(stack_bool[0,:,:,0].shape) # compensation events matrix
    # thrs_matrix = np.zeros(stack_bool[0,:,:,0].shape) # under threshold events matrix
    
    for d in range(0, stack.shape[0]-1):
        
        # INITIALIZE ARRAYS
        comp_array=[] # Initialize array where to append the number of pixel that experienced compensation
        thrs_array=[] # Initialize array ehere to append the number of pixel that experienced threshold bias
        
        comp_matrix = np.zeros(stack_bool[0,:,:,0].shape) # Compensation events matrix
        thrs_matrix = np.zeros(stack_bool[0,:,:,0].shape) # Under threshold events matrix
        
        for t in range(0, stack.shape[0]-1):
            DoD_envelope = np.nansum(abs(stack_bool[:t+2,:,:,d]), axis=0) # Perform the DoD envelope
            DoD_envelope = np.where(DoD_envelope>0, 1, DoD_envelope) # Make the envelope of DoD as a boolean map
            DoD_boundary = abs(stack_bool[0,:,:,t+1]) # Extract the DoD within at which the envelope take place
            diff_matrix = DoD_envelope - DoD_boundary # Perform the difference between the DoDs_envelope and the boundary DoD
            
            # Partial maps of compensation and under threshold effects
            comp_matrix += diff_matrix==1
            thrs_matrix += diff_matrix==-1
        
            comp = np.nansum(diff_matrix==1) # Compute the number of pixel that experience compensation
            thrs = np.nansum(diff_matrix==-1) # Compute the number of pixel that experienced threshold bias
            comp_array = np.append(comp_array, comp) # Append values into the array
            thrs_array = np.append(thrs_array, thrs) # Append values into the array
            
        # Save array in txt
        np.savetxt(os.path.join(report_dir, run, run + '_' + str(d+1) + '_comp_analysis.txt'), comp_array, delimiter='\t')
        np.savetxt(os.path.join(report_dir, run, run + '_' + str(d+1) + '_thrs_analysis.txt'), thrs_array, delimiter='\t')
        
        # Reshape maps to get the right proportion of 1:10 in pixel dimension
        comp_matrix_rsh = np.repeat(comp_matrix, 10, axis=1)
        thrs_matrix_rsh = np.repeat(thrs_matrix, 10, axis=1)
        
        
        comp_matrix_rsh = np.array(comp_matrix_rsh, dtype=np.uint8)
        thrs_matrix_rsh = np.array(thrs_matrix_rsh, dtype=np.uint8)
        np.save(os.path.join(report_dir, run, run + '_' + str(d+1) + '_cumulative_compensation_effects.npy'), comp_matrix)
        np.save(os.path.join(report_dir, run, run + '_' + str(d+1) + '_cumulative_threshold_effects.npy'), thrs_matrix)
        
        # Save compensation and threshold event spatially distributed as maps
        imageio.imwrite(os.path.join(report_dir, run, run + '_' + str(d+1) + '_cumulative_compensation_effects.tiff'), comp_matrix_rsh)
        imageio.imwrite(os.path.join(report_dir, run, run + '_' + str(d+1) + '_cumulative_threshold_effects.tiff'), thrs_matrix_rsh)

    
    # Plot the results
    x_data = np.linspace(1,4.5,8)
    plt.plot(x_data, comp_array, label=f'comp {run}')
    
# Add labels and a legend
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.legend()

# Show the plot (optional)
plt.show()
    

end = time.time()
print()
print('Execution time: ', (end-start), 's')